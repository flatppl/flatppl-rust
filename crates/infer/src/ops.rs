//! The per-op type/phase rule catalogue (engine-concepts §18: one source of
//! truth per op — this table is what later passes share).
//!
//! Coverage is incremental and honest: ops without a rule yield `%deferred`
//! plus a once-per-op note (see crate docs). Rules mirror the spec tables —
//! §07 functions (domains/results), §08 distributions (variate domains),
//! §06 measure combinators, §04 reified callables.

use flatppl_core::{
    Call, CallHead, Dim, Inputs, Mass, Node, NodeId, Phase, Ref, RefNs, Scalar, ScalarType, Symbol,
    Type, ValueSet,
};

use crate::Level;
use crate::consteval::{count_dims, resolve_dim, static_dim};
use crate::trace::{Inferencer, join_phase};

/// `(node, type, phase)` of an inferred positional argument.
type ArgInfo = (NodeId, Type, Phase);
/// `(name, node, type, phase)` of an inferred named argument.
type NamedInfo = (Symbol, NodeId, Type, Phase);

pub(crate) fn literal_type(s: &Scalar) -> Type {
    match s {
        Scalar::Int(_) => Type::Scalar(ScalarType::Integer),
        Scalar::Real(_) => Type::Scalar(ScalarType::Real),
        Scalar::Bool(_) => Type::Scalar(ScalarType::Boolean),
        // Strings have no FlatPIR value type (paths / field names — metadata);
        // `Any` keeps them neutral in joins, and literals never emit `%meta`.
        Scalar::Str(_) => Type::Any,
    }
}

/// Built-in constants in value position. Sets and ops-as-values have no
/// first-class `Type` (they are resolved structurally where consumed:
/// `elementof` reads its set argument's *node*, `broadcast` its head).
pub(crate) fn const_type(name: &str) -> Type {
    match name {
        "pi" | "inf" => Type::Scalar(ScalarType::Real),
        "im" => Type::Scalar(ScalarType::Complex),
        _ => Type::Any,
    }
}

/// Dispatch a call to its op rule. `joined` is the §04 ancestor-rule phase
/// join over all inputs; rules override it only where the op itself
/// introduces a phase (`elementof`, `draw`, reification closure, loaders).
pub(crate) fn call_rule(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    call: &Call,
    callee: Option<(NodeId, Type)>,
    args: &[ArgInfo],
    named: &[NamedInfo],
    joined: Phase,
) -> (Type, Phase) {
    // User-defined callable application: the result looks through the callee
    // to the reified body (spec §11 reified callables).
    if let Some((callee_node, callee_ty)) = callee {
        // `true`/`false` are lexed unconditionally to a bool `Lit` (never a
        // `Ref`), so `true(0.5)` is postfix APPLICATION on that literal (spec
        // §11), reaching this `CallHead::User` path instead of the
        // `CallHead::Builtin` fallback below that types every other
        // predefined constant's application `Type::Failed`. Without this,
        // `callee_ty` here is a bare `Scalar(Boolean)` — matching neither
        // `Type::Function` nor `Type::Kernel` in `user_call_type` — so it fell
        // through to that function's honest `Type::Deferred` default, silent
        // and indistinguishable from "no rule yet".
        if let Node::Lit(Scalar::Bool(b)) = inf.module.node(callee_node) {
            let name = if *b { "true" } else { "false" };
            inf.diags.push(crate::Diagnostic::error_at(
                id,
                format!(
                    "`{name}` is a predefined constant (spec §03), not a callable, \
                     so it cannot be applied to arguments (spec §04 \"Language \
                     design\": no callable has nullary inputs, which is what a \
                     known value like `{name}` would need to be one)"
                ),
            ));
            return (
                Type::Failed(format!("{name} is not callable").into()),
                Phase::Fixed,
            );
        }
        if let Some(ty) = user_arity_check(inf, id, callee_node, &callee_ty, args, named) {
            return (ty, joined);
        }
        let ty = user_call_type(inf, callee_node, &callee_ty, args, named);
        return (ty, joined);
    }

    let flatppl_core::CallHead::Builtin(op) = call.head else {
        unreachable!("user calls handled above");
    };
    let name = inf.module.resolve(op).to_string();

    // Reified callables (`functionof` / `kernelof`) — typed by their boundary
    // + body, and always *fixed* (a reification closes over its ancestry).
    if call.inputs.is_some() {
        return (reification_type(inf, id, call, &name, args), Phase::Fixed);
    }

    // Keyword arguments moved into their DECLARED POSITIONS, once, before any rule reads
    // them — §04 "Calling conventions" makes the two spellings the same call:
    //
    //   "All built-in ordinary callables have a defined input order and accept both
    //    positional and keyword arguments."
    //
    // The per-op rules below index fixed positions (`args.get(1)`, `args.first()`), so
    // without this a keyword call reached them with an EMPTY `args` and every
    // position-indexed check silently deferred. `named` is deliberately left intact: the
    // handful of rules that read it resolve a parameter by name and pick one value, so
    // seeing it in both places changes nothing, and the rules whose variadic inputs are
    // genuinely named (`record`, `table`, `joint`, `jointchain`, `broadcast`) declare no
    // parameter names at all and so are never normalized.
    // Call arity, where the catalogue declares a parameter list. Ahead of the
    // per-op rules because most of them index fixed argument positions and so
    // ignore extras and silently type an under-supplied call.
    if let Some(ty) = arity_check(inf, id, &name, args, named) {
        return (ty, joined);
    }

    // AFTER `arity_check`, which must see the call as WRITTEN: it counts
    // `args.len() + named.len()` and resolves the splat, so handing it a vector holding the
    // keywords as well as `named` would double every keyword. Normalization is for POSITIONS
    // only; counts and names are settled above.
    let normalized;
    let args = match normalize_keyword_args(inf.module, &name, args, named) {
        Some(v) => {
            normalized = v;
            normalized.as_slice()
        }
        None => args,
    };

    let ty = match name.as_str() {
        // ---- arithmetic (spec §07) — structural: result depends on arg shapes/types ----
        "add" | "sub" => elementwise2(&args.first(), &args.get(1)),
        "mul" => mul_type(args),
        "pow" => promote2(arg_ty(args, 0), arg_ty(args, 1)),
        "neg" => args.first().map_or(Type::Deferred, |(_, t, _)| t.clone()),
        "min" | "max" | "atan2" => promote2(arg_ty(args, 0), arg_ty(args, 1)),
        // `divide(a, b) = a / b` over real OR complex scalars (spec §07):
        // complex if either operand is complex, else real (true division —
        // integer operands still divide to real). NOT a constant Scalar(Real):
        // the complex case must promote. See `divide_type`.
        "divide" => divide_type(arg_ty(args, 0), arg_ty(args, 1)),

        // `div(a, b) = ⌊a/b⌋` and `mod(a, b) = a − b·⌊a/b⌋` are integer-domain
        // (spec §07: operands and result `integers`, `b ≠ 0`). The result is
        // always `integer` (per the catalogue); additionally reject a real or
        // complex operand as a static error rather than letting it through to a
        // silently-fractional value — real division is the separate `divide`
        // op. Booleans embed into integers (spec §03 `booleans ⊂ integers ⊂
        // reals`) and pass; deferred/any operands defer to runtime.
        "div" | "mod" => {
            let mut ok = true;
            for i in 0..2 {
                let Some(t) = arg_ty(args, i) else { continue };
                if let Some(kind @ (ScalarType::Real | ScalarType::Complex)) = scalar_kind(t) {
                    let anchor = args.get(i).map_or(id, |(n, _, _)| *n);
                    let hint = if name == "div" {
                        " — use `divide` for real division"
                    } else {
                        ""
                    };
                    inf.diags.push(crate::Diagnostic::error_at(
                        anchor,
                        format!(
                            "`{name}` is integer-domain (spec §07): argument {} \
                             is {kind}, but `{name}` requires integers{hint}",
                            i + 1,
                        ),
                    ));
                    ok = false;
                }
            }
            if ok {
                Type::Scalar(ScalarType::Integer)
            } else {
                Type::Failed(format!("{name} non-integer operand").into())
            }
        }

        // ---- containers (spec §03) — structural: result type threads arg types ----
        "vector" => vector_type(inf, args),
        "tuple" => Type::Tuple(args.iter().map(|(_, t, _)| t.clone()).collect()),
        // `record(t)` auto-splats a single table into a record of its column
        // vectors (spec §03); otherwise a record of its named fields.
        // `record(r)` on an argument that is ALREADY a record is the
        // same-kind shape §03/§04 auto-splatting never sanctions (that rule
        // converts the OTHER aggregate kind, never a value into its own
        // kind) — refused rather than silently returning an empty record
        // (the old `_` arm's `named` was empty here, since the call is
        // positional). See [`refuse_same_kind_constructor`].
        "record" => match (named.is_empty(), args) {
            (true, [(_, Type::Table { columns, nrows }, _)]) => record_from_table(columns, *nrows),
            (true, [(_, Type::Record(_), _)]) => refuse_same_kind_constructor(inf, id, "record"),
            _ => Type::Record(named.iter().map(|(n, _, t, _)| (*n, t.clone())).collect()),
        },
        // `table(r)` auto-splats a single record-of-vectors into columns (spec
        // §03); otherwise a table of its named columns. `table(t)` on an
        // argument already a table is the same same-kind shape as `record`
        // above — refused rather than silently deferring.
        "table" => match (named.is_empty(), args) {
            (true, [(node, Type::Record(fields), _)]) => {
                let cols: Vec<(Symbol, &Type, NodeId)> =
                    fields.iter().map(|(n, t)| (*n, t, *node)).collect();
                build_table(inf, &cols)
            }
            (true, [(_, Type::Table { .. }, _)]) => refuse_same_kind_constructor(inf, id, "table"),
            _ => table_type(inf, named),
        },
        "rowstack" => rowstack_type(arg_ty(args, 0)),
        "get" => get_type(inf, args, /*base=*/ 1),
        "get0" => get_type(inf, args, /*base=*/ 0),
        // §07 "Operator-equivalent functions" gives every comparison a SCALAR
        // domain, so an array operand has no §07 meaning — see
        // [`refuse_array_comparison`], which is where the citation lives.
        "equal" | "unequal" | "lt" | "le" | "gt" | "ge"
            if args.iter().any(|(_, t, _)| is_array_like(t))
                || named.iter().any(|(_, _, t, _)| is_array_like(t)) =>
        {
            refuse_array_comparison(inf, id, &name, args, named)
        }
        "indicesof" | "indicesof0" => Type::Array {
            shape: Box::new([Dim::Dynamic]),
            elem: Box::new(Type::Scalar(ScalarType::Integer)),
        },
        // §07 "Table reductions": applied to a TABLE these reduce column-wise to a
        // record. Guarded so every non-table argument keeps the arm it had —
        // `sum`/`mean`/`prod` the array rule just below, `var`/`std` their catalogue
        // row (`result: Scalar(Real)`), `maximum`/`minimum`/`median`/`lany`/`lall`
        // their catalogue row, each via the `_` arm. See [`table_reduction_type`].
        "sum" | "mean" | "var" | "std" | "prod" | "maximum" | "minimum" | "median" | "lany"
        | "lall"
            if matches!(arg_ty(args, 0), Some(Type::Table { .. })) =>
        {
            table_reduction_type(&name, arg_ty(args, 0))
        }
        // `sum` / `prod` / `mean` reduce a real/complex array to its element
        // type (spec §07 Reductions): mean of a complex array is complex.
        // (NOT a constant Scalar(Real); legacy ops.rs returned Real always.)
        "sum" | "prod" | "mean" => reduce_type(&name, arg_ty(args, 0)),
        // §03's boolean promotion reaches the CUMULATIVE pair for the same reason it
        // reaches `sum`: `cumsum([true, true, false])` is `[1, 2, 2]`, and `2` is not a
        // boolean, so the catalogue's `SameAsArg(0)` row typed the result
        // `cartpow(booleans, 3)` — a set that does not contain the value. Guarded, so
        // every non-boolean element keeps that row via the catalogue arm below.
        "cumsum" | "cumprod" if bool_elem_array(arg_ty(args, 0)) => {
            cumulative_bool_type(arg_ty(args, 0))
        }
        // Vector normalizations: same-shape real vector — shape must thread through.
        "softmax" | "logsoftmax" | "l1unit" | "l2unit" => match arg_ty(args, 0) {
            Some(Type::Array { shape, .. }) if shape.len() == 1 => Type::Array {
                shape: shape.clone(),
                elem: Box::new(Type::Scalar(ScalarType::Real)),
            },
            _ => Type::Deferred,
        },

        // ---- value-preserving assertion (spec §07) ----
        // `checked`/`fixed` are identity for typing (spec §03: `fixed(x)` ≡
        // `identity(x)`, a tooling hint) — the wrapped value's type rides through.
        // (`identity` itself, `ifelse`, `real`, `imag` are catalogue rows —
        // SameAsArg / CommonOf / RealOfArgShape.)
        "checked" | "fixed" => args.first().map_or(Type::Deferred, |(_, t, _)| t.clone()),

        // ---- value-shaped array constructors (spec §07) ----
        // Structural, NOT catalogue rows: the result RANK comes from a `size`
        // argument's value (`zeros(3)` is a vector, `zeros([2, 3])` a matrix),
        // which a single catalogue row cannot express. `count_dims` reads the
        // size arg's shape (vector literal → one dim per element, else a single
        // dim), resolving fixed-integer dims at Level::Shape (§17.1).
        // `zeros`/`ones` are real-valued; `fill(x, size)` takes x's element kind.
        "zeros" | "ones" => Type::Array {
            shape: args.first().map_or_else(
                || Box::new([Dim::Dynamic]) as Box<[Dim]>,
                |a| count_dims(inf, a.0),
            ),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        },
        "fill" => Type::Array {
            shape: args.get(1).map_or_else(
                || Box::new([Dim::Dynamic]) as Box<[Dim]>,
                |a| count_dims(inf, a.0),
            ),
            elem: Box::new(match arg_ty(args, 0) {
                Some(Type::Scalar(s)) => Type::Scalar(*s),
                _ => Type::Scalar(ScalarType::Real),
            }),
        },
        // array(data, size, dimorder): n-d array of `size`, element kind from data.
        "array" => Type::Array {
            shape: args.get(1).map_or_else(
                || Box::new([Dim::Dynamic]) as Box<[Dim]>,
                |a| count_dims(inf, a.0),
            ),
            elem: Box::new(Type::Scalar(
                arg_ty(args, 0)
                    .and_then(|t| match t {
                        Type::Scalar(s) => Some(*s),
                        Type::Array { .. } => elem_scalar_kind_of(t),
                        _ => None,
                    })
                    .unwrap_or(ScalarType::Real),
            )),
        },
        // tile(A, size) keeps A's rank and element kind; only the sizes change.
        "tile" => arg_ty(args, 0).map_or(Type::Deferred, with_dynamic_dims),
        // `aggregate(f, output_axes, expr)` / `metricsum(metric, output_axes,
        // expr)` (spec §04): an einsum-style reduction. The result RANK is the
        // number of output axes (the `output_axes` vector at arg 1); the element
        // kind comes from the reduced `expr` (arg 2). Empty output axes → a
        // scalar (e.g. `aggregate(sum, [], A[.i]*B[.i])`). A non-literal axis
        // list leaves the rank unknown → defer.
        "aggregate" | "metricsum" => {
            let elem = Type::Scalar(aggregate_result_kind(inf, args).unwrap_or_else(|| {
                arg_ty(args, 2)
                    .and_then(elem_scalar_kind_of)
                    .unwrap_or(ScalarType::Real)
            }));
            match args.get(1).and_then(|a| output_axis_names(inf, a.0)) {
                // No output axes → full contraction → scalar.
                Some(axes) if axes.is_empty() => elem,
                Some(axes) => {
                    // Exact dims: trace each output axis to the input dim it
                    // indexes in the body (`A[.i, .j]` → `.i` is A's flat dim 0).
                    let mut extents = std::collections::HashMap::new();
                    if let Some(b) = args.get(2) {
                        collect_axis_dims(inf, b.0, &mut extents);
                    }
                    let dims: Vec<Dim> = axes
                        .iter()
                        .map(|a| extents.get(a).copied().unwrap_or(Dim::Dynamic))
                        .collect();
                    Type::Array {
                        shape: dims.into_boxed_slice(),
                        elem: Box::new(elem),
                    }
                }
                None => Type::Deferred,
            }
        }
        // reduce(f, xs) folds xs with an associative f; spec §07 requires f to
        // return the element type of xs, so the result IS that element type
        // (a vector of reals reduces to a real, a vector of vectors to a vector).
        "reduce" => match arg_ty(args, 1) {
            Some(Type::Array { elem, .. }) => (**elem).clone(),
            _ => Type::Deferred,
        },
        // filter(pred, data) keeps a subset of data's elements/rows: same type
        // and rank as data, with the filtered axis now dynamic.
        "filter" => arg_ty(args, 1).map_or(Type::Deferred, with_dynamic_dims),
        // partition(xs, spec) splits a vector into a vector of sub-vectors (spec
        // §07): an outer vector whose elements are dynamic-length copies of xs.
        "partition" => match arg_ty(args, 0) {
            Some(t @ Type::Array { .. }) => Type::Array {
                shape: Box::new([Dim::Dynamic]),
                elem: Box::new(with_dynamic_dims(t)),
            },
            _ => Type::Deferred,
        },
        // selectbins(edges, region, counts) returns a shorter count array (spec
        // §07): counts' type and rank, with the selected axis dynamic.
        "selectbins" => arg_ty(args, 2).map_or(Type::Deferred, with_dynamic_dims),
        // addaxes(A, n_leading, n_trailing) (spec §07) inserts `n_leading`
        // size-1 axes before A's axes and `n_trailing` after — exact when the
        // counts are fixed integers: result shape = [1;nl] ++ A.shape ++ [1;nt],
        // element preserved. (e.g. A:(3,4,5), addaxes(A,2,3) → (1,1,3,4,5,1,1,1).)
        "addaxes" => {
            // The counts are non-negative fixed integers; `resolve_dim` folds
            // them (a `Static(n)` is ≥ 0 by construction) AND emits the loud
            // op-gap diagnostic if a count uses an unfoldable fixed op (§17.1).
            let nl = args.get(1).map(|a| resolve_dim(inf, a.0));
            let nt = args.get(2).map(|a| resolve_dim(inf, a.0));
            match (arg_ty(args, 0), nl, nt) {
                (
                    Some(Type::Array { shape, elem }),
                    Some(Dim::Static(nl)),
                    Some(Dim::Static(nt)),
                ) => {
                    let mut dims: Vec<Dim> =
                        std::iter::repeat_n(static_dim(1), nl as usize).collect();
                    dims.extend_from_slice(shape);
                    dims.extend(std::iter::repeat_n(static_dim(1), nt as usize));
                    Type::Array {
                        shape: dims.into_boxed_slice(),
                        elem: elem.clone(),
                    }
                }
                _ => Type::Deferred,
            }
        }
        // splitblocks(A, blocksize) (spec §07) nests A into a vector of equal
        // sub-arrays. Exact for a 1-D scalar vector → vector of sub-vectors;
        // multi-D outer rank is value-dependent, so those defer.
        "splitblocks" => match arg_ty(args, 0) {
            Some(Type::Array { shape, elem })
                if shape.len() == 1 && matches!(elem.as_ref(), Type::Scalar(_)) =>
            {
                Type::Array {
                    shape: Box::new([Dim::Dynamic]),
                    elem: Box::new(Type::Array {
                        shape: Box::new([Dim::Dynamic]),
                        elem: elem.clone(),
                    }),
                }
            }
            _ => Type::Deferred,
        },
        // `cat(x, y, …)` concatenates same-shape-class values (spec §07): all
        // scalars → a length-n vector, all 1-D vectors → one concatenated
        // vector, all records → a merged record. The single `cat` shape rule —
        // shared with positional `cartprod` / `joint`; mixing shape classes is a
        // static error.
        "cat" => {
            let parts: Vec<Type> = args.iter().map(|(_, t, _)| t.clone()).collect();
            cat_or_diagnose(inf, id, "cat", &parts)
        }

        // ---- parameters / inputs (spec §04) ----
        "elementof" | "external" => set_element_type(inf, args.first().map(|a| a.0)),
        // `load_data(source, valueset)` (spec §07 `load_data`): "`valueset`
        // fully determines the result's shape" — "A scalar set yields a scalar,
        // `cartpow` an array, `cartprod` a record, and a power of a record set a
        // table". So the result is a MEMBER of the declared set, exactly as for
        // `elementof`/`external`, with no extra row axis: nothing reads the
        // source to discover a shape. `valueset` is the keyword or the second
        // positional arg (after `source`).
        "load_data" => {
            let vs = named_or_positional_node(inf.module, named, args, "valueset", 1);
            set_element_type(inf, vs)
        }

        // ---- measure algebra (spec §06) ----
        "lawof" => match lawof_mass_gate(inf, args) {
            Some(failed) => failed,
            None => lawof_type(args.first().map(|(_, t, _)| t)),
        },
        "draw" => match draw_mass_gate(inf, args) {
            Some(failed) => failed,
            None => measure_domain(arg_ty(args, 0)),
        },
        "iid" => iid_type(inf, args),
        // Measure-transforming ops keep the domain but get a FRESH mass slot
        // — their total mass differs from the base's and is computed by the
        // normalization-level rules (inheriting it via the type clone would
        // smuggle the base's class through `fill_mass`).
        "truncate" | "normalize" => fresh_measure(arg_ty(args, 0)),
        // Domain-preserving measure-algebra ops (spec §06): the result is a
        // measure over the SAME value domain as its measure argument, with a
        // fresh (recomputed) mass — like truncate/normalize.
        //   `restrict(M, S)`   — restrict M to S
        //   `superpose(M1, …)` — measure addition M1 + M2 + … (shared domain)
        //   `locscale(M, …)`   — affine pushforward x → scale·x + shift
        // These no longer defer: even before the engine evaluates their mass,
        // the value domain is known, so the type slot carries `(%measure …)`.
        "restrict" | "locscale" => fresh_measure(arg_ty(args, 0)),
        // EVERY `superpose` argument must itself be a measure (spec §06: measure
        // addition) — `fresh_measure` otherwise passes a non-measure argument
        // straight through unchanged, so `superpose(record(m1 = n1, m2 = n2))`
        // typed as a RECORD of measures, with no diagnostic (`fresh_measure`'s
        // `Some(other) => other.clone()` arm). A measure position holding a
        // record is unambiguous, unlike the open same-kind-constructor ruling
        // above, so it is refused outright here rather than deferred to a card.
        // Checks the WHOLE argument list, not just position 0: a bad argument
        // anywhere else was silently DROPPED from the type entirely (worse than
        // position 0's silent pass-through) — `superpose(n, record(…))` typed as
        // `n`'s plain measure, `%unknown` mass, the record gone, no diagnostic.
        "superpose" => {
            let mut bad = false;
            for (node, t, _) in args {
                if let Some(kind) = non_measure_kind(t) {
                    inf.diags.push(crate::Diagnostic::error_at(
                        *node,
                        format!(
                            "`superpose`'s argument must be a measure (spec §06: \
                             measure addition); got {kind} instead"
                        ),
                    ));
                    bad = true;
                }
            }
            if bad {
                Type::Failed("superpose argument is not a measure".into())
            } else {
                fresh_measure(arg_ty(args, 0))
            }
        }
        "ksuperpose" => ksuperpose_type(inf, id, args, named),
        // `pushfwd(f, M)` (spec §06): a measure whose domain is the CODOMAIN of
        // `f`. `f` maps a value drawn from `M`, so binding its input to `M`'s
        // variate (domain + support value-set) and reading `f`'s body type gives
        // the codomain; fall back to `f`'s un-substituted body, then to `%any`
        // (honest — never a guess). Mass is filled (mass-preserving) by `fill_mass`.
        "pushfwd" => Type::Measure {
            domain: Box::new(pushfwd_codomain(inf, args).unwrap_or(Type::Any)),
            mass: Mass::Deferred,
        },
        // `markovchain(kernel, init, n)` / `kscan(kernel, init, xs)` (spec §06):
        // a measure over a length-`len` trajectory in `init`'s state space.
        // Domain is `array[len]` of `init`'s type — `n` (markovchain) folds at
        // Level::Shape, `lengthof(xs)` (kscan) is xs's leading dim. (Record-state
        // trajectories are tables — left with a deferred domain for now.) Mass is
        // filled from the kernel's class in `fill_mass`.
        "markovchain" => {
            let len = args.get(2).map_or(Dim::Dynamic, |a| resolve_dim(inf, a.0));
            trajectory_measure(arg_ty(args, 1), len)
        }
        "kscan" => {
            let len = match arg_ty(args, 2) {
                Some(Type::Array { shape, .. }) if !shape.is_empty() => shape[0],
                _ => Dim::Dynamic,
            };
            trajectory_measure(arg_ty(args, 1), len)
        }
        // `kchain(M, K1, …, Kn)` (spec §06): Kleisli bind — marginalizes the
        // intermediate variates, KEEPS THE LAST component's variate. Mass is
        // filled by `fill_mass`.
        "kchain" => Type::Measure {
            domain: Box::new(
                args.last()
                    .and_then(|(n, t, _)| component_variate(inf, *n, t))
                    .unwrap_or(Type::Deferred),
            ),
            mass: Mass::Deferred,
        },
        // `jointchain(M, K1, …)` (spec §06): dependent joint — KEEPS ALL variates
        // (`cat` of every component's, or a named record in keyword form). Mass
        // is filled by `fill_mass`.
        "jointchain" => Type::Measure {
            domain: Box::new(jointchain_domain(inf, args, named)),
            mass: Mass::Deferred,
        },
        // `scan(f, init, xs)` (spec §04) is the DETERMINISTIC left scan — a value,
        // not a measure: `array[lengthof(xs)]` of the accumulator type (= init's
        // type). The stochastic analogue is `kscan`.
        "scan" => match arg_ty(args, 1) {
            Some(t @ (Type::Scalar(_) | Type::Array { .. })) => {
                let len = match arg_ty(args, 2) {
                    Some(Type::Array { shape, .. }) if !shape.is_empty() => shape[0],
                    _ => Dim::Dynamic,
                };
                Type::Array {
                    shape: Box::new([len]),
                    elem: Box::new(t.clone()),
                }
            }
            _ => Type::Deferred,
        },
        // `fchain(f1, f2, …)` (spec §04) composes deterministic functions; the
        // result is a function with `f1`'s input signature (output type is not
        // tracked by `Type::Function`).
        "fchain" => match arg_ty(args, 0) {
            Some(Type::Function { inputs }) => Type::Function {
                inputs: inputs.clone(),
            },
            _ => Type::Deferred,
        },
        // `disintegrate(selector, joint)` (spec §06) splits a joint measure into
        // a `(forward_kernel, marginal)` tuple. When the joint is a record-domain
        // measure and the selector is a static set of field names, the marginal
        // carries the complement fields and the kernel inputs are those complement
        // (conditioning) variates. See `disintegrate_type` for the full logic.
        "disintegrate" => disintegrate_type(inf, call, args),
        // `relabel(M, labels)` (spec §06) renames the variate; the value domain
        // AND total mass are unchanged, so the measure type passes through whole
        // (unlike normalize/truncate, which reset the mass slot).
        "relabel" => arg_ty(args, 0).cloned().unwrap_or(Type::Deferred),
        // `weighted(weight, base)` / `logweighted(logweight, base)` (spec
        // §06): the measure is the SECOND argument.
        "weighted" | "logweighted" => fresh_measure(arg_ty(args, 1)),
        // Reference measures (spec §06): measures over their support set. The
        // `support` set is given as the named kwarg, auto-splatted from a
        // positional `record(support = S)` (§04), or as the plain positional set.
        "Lebesgue" | "Counting" => {
            let support_node = lebesgue_counting_support_node(inf, args, named);
            Type::Measure {
                domain: Box::new(set_element_type(inf, support_node)),
                mass: Mass::Deferred,
            }
        }
        // `Dirac(value = v)` (spec §06) is the point-mass probability measure at
        // `v`, for any variate type: the domain is `v`'s type. `value` is given
        // as the named kwarg (spec form), auto-splatted from a positional
        // `record(value = v)` (§04 — so `Dirac(record(value = v))` is a point
        // mass at `v`, NOT at the record), or as a plain positional value. Mass
        // is normalized (total mass 1) — set in `fill_mass`.
        "Dirac" => {
            let v = named
                .iter()
                .find(|(n, _, _, _)| inf.module.resolve(*n) == "value")
                .map(|(_, _, t, _)| t.clone())
                .or_else(|| {
                    splat_field(inf, args, named, "value").and_then(|n| inf.lookup_type(n).cloned())
                })
                .or_else(|| arg_ty(args, 0).cloned());
            match v {
                Some(t) => Type::Measure {
                    domain: Box::new(t),
                    mass: Mass::Deferred,
                },
                None => Type::Deferred,
            }
        }
        // `bayesupdate(L, prior)` (spec §06): the unnormalized posterior is a
        // measure over the prior's domain — pick the measure-typed argument,
        // with a fresh mass slot (the posterior's mass is the evidence).
        "bayesupdate" => fresh_measure(
            args.iter()
                .map(|(_, t, _)| t)
                .find(|t| matches!(t, Type::Measure { .. })),
        ),
        "joint" => joint_type(inf, id, args, named),
        "likelihoodof" => likelihood_type(inf, args),
        "joint_likelihood" => joint_likelihood_type(args),

        // ---- explicit RNG (spec §07) ----
        "rnginit" => Type::RngState,
        "rand" => match rand_mass_gate(inf, args) {
            Some(failed) => failed,
            None => match measure_domain(arg_ty(args, 1)) {
                Type::Deferred => Type::Deferred,
                domain => Type::Tuple(Box::new([domain, Type::RngState])),
            },
        },

        // ---- measure-kernel evaluation primitives (spec §07 sec:measure-eval-prims) ----
        // FlatPDL primitive surface; TYPE-LEVEL ONLY (flatppl-rust does not evaluate
        // densities/samples). `builtin_logdensityof` is a real scalar (scalar-over-batch,
        // engine-concepts §13.3), independent of the kernel's variate; `-inf` outside
        // support is a runtime value, not a type concern.
        "builtin_logdensityof" => Type::Scalar(ScalarType::Real),
        // `builtin_sample(rngstate, kernel, kernel_input, n, m, …)` → `(variate,
        // new_rngstate)`. Kernel = arg 1, kernel_input = arg 2. The variate comes from
        // `component_variate` (reified kernels — the accessor `kchain` uses) or, for a
        // bare distribution constructor, from `kernel_variate` (the catalogue). Any
        // trailing `n, m, …` args (arg 3 on) are size dims: per spec §07, they array-ify
        // the variate into an IID array of shape `(n, m, …)` (no dims → the bare variate,
        // unchanged from before).
        "builtin_sample" => {
            let k = args.get(1);
            let variate = k
                .and_then(|(n, t, _)| component_variate(inf, *n, t))
                .or_else(|| {
                    k.and_then(|(n, _, _)| kernel_variate(inf, *n, args.get(2).map(|a| a.0)))
                });
            match variate {
                Some(v) => {
                    let size_args = args.get(3..).unwrap_or(&[]);
                    let variate = if size_args.is_empty() {
                        v
                    } else {
                        Type::Array {
                            shape: size_args
                                .iter()
                                .map(|(n, _, _)| resolve_dim(inf, *n))
                                .collect(),
                            elem: Box::new(v),
                        }
                    };
                    Type::Tuple(Box::new([variate, Type::RngState]))
                }
                None => non_kernel_or_defer(inf, k, "builtin_sample", "argument 2"),
            }
        }
        // The four transports `f(kernel, kernel_input, x)` → the kernel's variate.
        // Kernel = arg 0, kernel_input = arg 1. Same kernel resolution as
        // `builtin_sample` (reified kernel, then bare constructor via the catalogue).
        // (The discrete-kernel transport refusal — §07 "use of an undefined transport
        // function is a static error" — is a follow-up; v1 types the variate regardless.)
        "builtin_touniform" | "builtin_fromuniform" | "builtin_tonormal" | "builtin_fromnormal" => {
            let k = args.first();
            let variate = k
                .and_then(|(n, t, _)| component_variate(inf, *n, t))
                .or_else(|| {
                    k.and_then(|(n, _, _)| kernel_variate(inf, *n, args.get(1).map(|a| a.0)))
                });
            match variate {
                Some(v) => v,
                None => non_kernel_or_defer(inf, k, name.as_str(), "argument 1"),
            }
        }

        // ---- multi-file (deferred — see TODO) ----
        "load_module" | "standard_module" => Type::Module,

        // ---- set constructors (spec §03) — set objects have no first-class
        // type; consumers (`elementof`, `truncate`, …) read them structurally.
        "interval" | "cartprod" => Type::Any,
        // `cartpow(S, size)` takes exactly a set and a size; the size is an
        // integer (1-D) or a vector of positive integers (multi-axis), §03
        // "Cartesian power". The legacy variadic `cartpow(S, d1, d2, …)` form
        // is not in the spec — reject it rather than silently reading only the
        // first dimension (a multi-axis power is `cartpow(S, [d1, d2, …])`).
        "cartpow" => {
            if args.len() == 2 {
                Type::Any
            } else {
                inf.diags.push(crate::Diagnostic::error_at(
                    id,
                    "`cartpow` takes a set and a size: `cartpow(S, n)` or, for a \
                     multi-axis power, `cartpow(S, [d1, d2, …])` (a single vector \
                     size). The variadic `cartpow(S, d1, d2, …)` form is not valid \
                     (spec §03).",
                ));
                Type::Failed("cartpow expects (set, size)".into())
            }
        }

        // ---- broadcasting (spec §04) ----
        "broadcast" => broadcast_type(inf, args, named),

        // ---- catalogue dispatch (spec §07 functions + spec §08 distributions) ----
        // Per-name functions whose result is a pure scalar (constant, RealOrComplexOfArg,
        // or DomainMap) are declared in catalogue.ron and lowered here.
        // Distribution constructors (Sig::Distribution rows) are also dispatched here.
        // Structural ops above cannot be expressed in ResultSig and stay as code.
        _ => match function_result(inf.module, &name, args) {
            Some(ty) => ty,
            None => match distribution_domain(inf, &name, args, named) {
                Some(domain) => Type::Measure {
                    domain: Box::new(domain),
                    mass: Mass::Deferred,
                },
                None => {
                    // `%deferred` means "no type rule YET" and is honest only
                    // for a name that IS a built-in. A head naming nothing in
                    // the `base` namespace resolves nowhere, and §04 "Name
                    // resolution" makes that a static error — the call-head half
                    // of the same rule the bare-atom arm enforces in `trace.rs`.
                    // Without this, `y = nromal(1.0)` inferred with only a note
                    // and the determiniser emitted the free call verbatim.
                    //
                    // User callables and §09 `alias.member` calls are
                    // `CallHead::User` and returned far above, so they never
                    // reach here.
                    if !crate::builtins::is_base_name(&name) {
                        inf.diags.push(crate::Diagnostic::error_at(
                            id,
                            format!(
                                "unresolvable call to `{name}`: not a binding in this module and \
                                 not a FlatPPL built-in (spec §04 \"Name resolution\")"
                            ),
                        ));
                        return (Type::Failed("unresolvable name".into()), Phase::Fixed);
                    }
                    // A predefined constant (spec §03) is a KNOWN VALUE, never a
                    // callable — §04 "no callables may have nullary inputs, as this
                    // would make them equivalent to known values". Without this,
                    // `pi(0.5)` (or `reals(0.5)`, `true(0.5)`, …) reached no rule
                    // above and fell through to the honest-gap arm below, typing
                    // `%deferred` with no diagnostic — indistinguishable from "no
                    // rule yet" and invisible to the `is_flatpdl` `Type::Failed`
                    // backstop.
                    if crate::builtins::is_predefined_constant(&name) {
                        inf.diags.push(crate::Diagnostic::error_at(
                            id,
                            format!(
                                "`{name}` is a predefined constant (spec §03), not a callable, \
                                 so it cannot be applied to arguments (spec §04 \"Language \
                                 design\": no callable has nullary inputs, which is what a \
                                 known value like `{name}` would need to be one)"
                            ),
                        ));
                        return (
                            Type::Failed(format!("{name} is not callable").into()),
                            Phase::Fixed,
                        );
                    }
                    inf.note_gap(op);
                    Type::Deferred
                }
            },
        },
    };

    let phase = match name.as_str() {
        "elementof" => Phase::Parameterized,
        "external" | "load_data" | "load_module" | "standard_module" => Phase::Fixed,
        "draw" => Phase::Stochastic,
        // `lawof` reifies a value into its law; the law is deterministic
        // (parameterized or fixed) — `lawof` absorbs the stochasticity of the
        // `draw` ancestors rather than propagating it (spec §04 "Phase of the
        // reified law"). Trace the argument's law-phase instead of inheriting
        // the stochastic `joined`.
        "lawof" => args
            .first()
            .map_or(Phase::Fixed, |a| law_phase(inf, a.0, 0)),
        _ => joined,
    };
    (ty, phase)
}

fn arg_ty(args: &[ArgInfo], i: usize) -> Option<&Type> {
    args.get(i).map(|(_, t, _)| t)
}

/// A positional argument vector with each keyword argument moved to its DECLARED position,
/// or `None` when the call needs no normalizing or cannot be normalized unambiguously.
///
/// §04 "Calling conventions": "All built-in ordinary callables have a defined input order and
/// accept both positional and keyword arguments." So `f(a = x, b = y)`, `f(x, y)` and the mixed
/// `f(x, b = y)` are the same call, and every rule that reads argument POSITIONS should see the
/// same vector for all three. §04 also fixes the mixed form's order — positional arguments bind
/// "in order", keyword arguments "by name" — so a positional prefix occupies `0..args.len()` and
/// each keyword lands at its own declared index.
///
/// **This is the one place the two spellings are reconciled.** The alternative — teaching each
/// per-op rule to look in both `args` and `named` — is two sites deriving one rule, which is how
/// the asymmetry it fixes arose in the first place: `arg_ty` reads only positions, so the
/// kernel-type check on the five `builtin_*` transports and the integer-domain check on
/// `div`/`mod` fired on `f(1.0, …)` and stayed silent on `f(rngstate = 1.0, …)`.
///
/// **Normalizes only an UNAMBIGUOUS mapping** — `None` for anything else, leaving the call
/// exactly as it arrived so the existing arity and name checks report it:
///
/// * no keyword arguments, or the row declares no parameter names (nothing to map by);
/// * a keyword whose name the row does not declare (the name check's business);
/// * a keyword targeting a position something else already fills — positionally, or by an
///   earlier keyword — a double-bound parameter. `arity_check` now calls
///   [`check_double_bound`] before this function ever runs and catches BOTH spellings of the
///   collision, so on a row with declared names the call already failed and returned there;
///   the two guards below (splat and mixed) are unreachable for such a row and stay only as
///   a defensive backstop should a nameless-row caller ever reach this branch;
/// * a GAP — a position neither spelling supplies — since a vector cannot carry a hole and
///   fabricating one would hide the under-supplied call the arity check exists to catch.
fn normalize_keyword_args(
    module: &flatppl_core::Module,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Vec<ArgInfo>> {
    let cat = crate::catalogue::builtin();
    let declared = cat.base_param_names(name)?;
    if named.is_empty() {
        // The SPLAT spelling. §04 makes `f(record(a = x, b = y))` "equivalent to
        // `f(a = x, b = y, ...)`", so it must reach the same per-op rules as the keyword form
        // it is defined as — which is the whole reason a splat inherited the keyword hole.
        // `arg_reading` splats for COUNTING and NAMING only; nothing rewrote `args`, so the
        // rules still saw one aggregate argument.
        //
        // The synthesized entries carry the AGGREGATE's node, not a per-field one: a
        // type-level splat has no per-field node (an opaque table has no field expressions),
        // and `supplied_arg_names` already anchors splat diagnostics at the aggregate for the
        // same reason. Types and positions are what the rules read.
        if cat.base_takes_aggregate_whole(name) {
            return None; // §04's single-input carve-out — no splat happens at all
        }
        let [(node, ty, phase)] = args else {
            return None;
        };
        let fields: &[(Symbol, Type)] = match ty {
            Type::Record(f) => f,
            Type::Table { columns, .. } => columns,
            _ => return None,
        };
        let mut slots: Vec<Option<ArgInfo>> = Vec::new();
        for (sym, fty) in fields {
            let supplied = module.resolve(*sym);
            let pos = declared.iter().position(|d| d.as_str() == supplied)?;
            if pos < slots.len() && slots[pos].is_some() {
                return None;
            }
            while slots.len() <= pos {
                slots.push(None);
            }
            slots[pos] = Some((*node, fty.clone(), *phase));
        }
        return slots.into_iter().collect();
    }
    let mut slots: Vec<Option<ArgInfo>> = Vec::new();
    // The positional prefix keeps its order (§04: "bound to the inputs in order").
    for a in args {
        slots.push(Some(a.clone()));
    }
    for (sym, node, ty, phase) in named {
        // `named` carries interned symbols; compare against the declared spelling.
        let supplied = module.resolve(*sym);
        let pos = declared.iter().position(|d| d.as_str() == supplied)?;
        if pos < slots.len() && slots[pos].is_some() {
            return None; // double-bound parameter — `check_double_bound` already caught this
        }
        while slots.len() <= pos {
            slots.push(None);
        }
        slots[pos] = Some((*node, ty.clone(), *phase));
    }
    // A hole means the call is under-supplied; hand it back unchanged for the arity check.
    slots.into_iter().collect()
}

/// The node supplied for a parameter that may be passed by keyword (`key = …`)
/// or positionally (index `pos`). Keyword takes precedence. Used by callables
/// whose args have both spellings (e.g. `load_data(source, valueset)`).
fn named_or_positional_node(
    module: &flatppl_core::Module,
    named: &[NamedInfo],
    args: &[ArgInfo],
    key: &str,
    pos: usize,
) -> Option<NodeId> {
    named
        .iter()
        .find(|(s, ..)| module.resolve(*s) == key)
        .map(|(_, n, ..)| *n)
        .or_else(|| args.get(pos).map(|(n, ..)| *n))
}

/// The scalar element kind of `t`, drilling through array nesting (an
/// elementwise op over an array carries the constraint to its elements).
/// `None` for non-scalar/non-array types (measures, modules, deferred,
/// failed, any) — those cannot be statically disproven as integer.
fn scalar_kind(t: &Type) -> Option<ScalarType> {
    match t {
        Type::Scalar(s) => Some(*s),
        Type::Array { elem, .. } => scalar_kind(elem),
        _ => None,
    }
}

/// Clone a measure type with its mass reset to `Deferred` (to be filled by
/// the normalization-level rule for the op at hand); non-measures clone
/// as-is, absent arguments defer.
fn fresh_measure(t: Option<&Type>) -> Type {
    match t {
        Some(Type::Measure { domain, .. }) => Type::Measure {
            domain: domain.clone(),
            mass: Mass::Deferred,
        },
        Some(other) => other.clone(),
        None => Type::Deferred,
    }
}

/// Numeric promotion: integer ⊔ integer = integer, real dominates integer,
/// complex dominates real; `Any` (placeholders) is absorbed.
fn promote2(a: Option<&Type>, b: Option<&Type>) -> Type {
    use ScalarType::*;
    let rank = |t: Option<&Type>| match t {
        Some(Type::Scalar(Integer)) | Some(Type::Scalar(Boolean)) => Some(0),
        Some(Type::Scalar(Real)) => Some(1),
        Some(Type::Scalar(Complex)) => Some(2),
        Some(Type::Any) => Some(-1),
        _ => None,
    };
    match (rank(a), rank(b)) {
        (Some(x), Some(y)) => match x.max(y) {
            -1 => Type::Any, // both unconstrained placeholders
            0 => Type::Scalar(Integer),
            1 => Type::Scalar(Real),
            _ => Type::Scalar(Complex),
        },
        _ => Type::Deferred,
    }
}

/// `reals, complexes` unary domain: complex in, complex out; else real.
fn real_or_complex(a: Option<&Type>) -> Type {
    match a {
        Some(Type::Scalar(ScalarType::Complex)) => Type::Scalar(ScalarType::Complex),
        _ => Type::Scalar(ScalarType::Real),
    }
}

/// `divide(a, b) = a / b` (spec §07): true division over scalars that are real
/// OR complex. The result is complex iff either operand is complex; otherwise
/// it is real — even for integer operands, since `1 / 2 = 0.5` is real (integer
/// floor-division is the separate `div` op). This differs from `promote2`,
/// which would keep integer/integer as integer.
fn divide_type(a: Option<&Type>, b: Option<&Type>) -> Type {
    use ScalarType::*;
    let is_complex = |t: Option<&Type>| matches!(t, Some(Type::Scalar(Complex)));
    let known_scalar = |t: Option<&Type>| {
        matches!(
            t,
            Some(Type::Scalar(Integer | Real | Complex | Boolean)) | Some(Type::Any)
        )
    };
    if is_complex(a) || is_complex(b) {
        Type::Scalar(Complex)
    } else if known_scalar(a) && known_scalar(b) {
        Type::Scalar(Real)
    } else {
        Type::Deferred
    }
}

/// `add`/`sub`: scalars promote; same-shape arrays go elementwise.
fn elementwise2(a: &Option<&ArgInfo>, b: &Option<&ArgInfo>) -> Type {
    let (a, b) = match (a, b) {
        (Some(a), Some(b)) => (&a.1, &b.1),
        _ => return Type::Deferred,
    };
    match (a, b) {
        (
            Type::Array {
                shape: sa,
                elem: ea,
            },
            Type::Array {
                shape: sb,
                elem: eb,
            },
        ) if sa == sb => Type::Array {
            shape: sa.clone(),
            elem: Box::new(promote2(Some(ea), Some(eb))),
        },
        _ => promote2(Some(a), Some(b)),
    }
}

/// `mul` (`a * b`, spec §07): scalar·scalar, scalar·array (both orders), and the
/// The spec-§07 `mul` matrix/vector forms: matrix·matrix (`[m,k]·[k,n] →
/// [m,n]`), matrix·vector (`[m,k]·[k] → [m]`), transposed-vector·vector (dot →
/// scalar), and vector·transposed-vector (outer → `[n,m]` matrix). A statically
/// provable dimension mismatch is a shape error (`%failed`). Matrices must be
/// FLAT rank-2 arrays (from `rowstack`/`colstack`); a nested vec-of-vec is not a
/// matrix (spec §03), so `mul` over those stays `%deferred` — correctly, not a gap.
fn mul_type(args: &[ArgInfo]) -> Type {
    let (a, b) = match (arg_ty(args, 0), arg_ty(args, 1)) {
        (Some(a), Some(b)) => (a, b),
        _ => return Type::Deferred,
    };
    let scalarish = |t: &Type| matches!(t, Type::Scalar(_) | Type::Any);
    match (a, b) {
        _ if scalarish(a) && scalarish(b) => promote2(Some(a), Some(b)),
        (Type::Array { .. }, s) if scalarish(s) => a.clone(),
        (s, Type::Array { .. }) if scalarish(s) => b.clone(),
        // Matrix multiply over flat rank-2 matrices: matrix·matrix → matrix,
        // matrix·vector → vector. The left operand is a rank-2 matrix; the right
        // is a matrix (rank-2) or a vector (rank-1). The shared inner dimension
        // (`sa[1]` vs the right's leading dim) must agree; the result drops it.
        (
            Type::Array {
                shape: sa,
                elem: ea,
            },
            Type::Array {
                shape: sb,
                elem: eb,
            },
        ) if sa.len() == 2 && (sb.len() == 2 || sb.len() == 1) => {
            if matches!((sa[1], sb[0]), (Dim::Static(k1), Dim::Static(k2)) if k1 != k2) {
                return Type::Failed(
                    "matrix multiply: inner dimensions disagree (spec §07)".into(),
                );
            }
            let out_shape: Box<[Dim]> = if sb.len() == 2 {
                Box::new([sa[0], sb[1]])
            } else {
                Box::new([sa[0]])
            };
            match promote2(Some(ea.as_ref()), Some(eb.as_ref())) {
                Type::Deferred => Type::Deferred, // non-numeric elements
                elem => Type::Array {
                    shape: out_shape,
                    elem: Box::new(elem),
                },
            }
        }
        // Transposed-vector · vector → scalar (inner / dot product). The lengths
        // must agree; a static mismatch is a shape error (spec §07).
        (
            Type::TVector { len: la, elem: ea },
            Type::Array {
                shape: sb,
                elem: eb,
            },
        ) if sb.len() == 1 => {
            if matches!((*la, sb[0]), (Dim::Static(k1), Dim::Static(k2)) if k1 != k2) {
                return Type::Failed("inner product: vector lengths disagree (spec §07)".into());
            }
            promote2(Some(ea.as_ref()), Some(eb.as_ref()))
        }
        // Vector · transposed-vector → matrix (outer product), shape `[n, m]`.
        (
            Type::Array {
                shape: sa,
                elem: ea,
            },
            Type::TVector { len: lb, elem: eb },
        ) if sa.len() == 1 => match promote2(Some(ea.as_ref()), Some(eb.as_ref())) {
            Type::Deferred => Type::Deferred,
            elem => Type::Array {
                shape: Box::new([sa[0], *lb]),
                elem: Box::new(elem),
            },
        },
        // Transposed-vector · matrix → transposed vector, `row[k] · [k, n] → row[n]`.
        // flatppl-design#77 (pending owner review) adds `transposed-vector–matrix` to
        // §07's `mul` row and states the result type in prose: "the product of a
        // transposed vector and a matrix is a transposed vector". So the result is a
        // ROW — `TVector`, not a `[1, n]` single-row matrix — which also keeps the
        // orientation-preserving pattern the inner (→ scalar) and outer (→ matrix)
        // products follow. The maths agrees: `(1×k)(k×n) = 1×n`.
        //
        // Ahead of the merged row: as of design `9e35262` §07 lists `matrix-vector`
        // and no `vector-matrix`. The mirror (`matrix · row`) stays `Deferred` and
        // #77 does not add it: `[m,k] · row[k]` does not conform for any `m, k` but
        // the degenerate `k = 1`.
        (
            Type::TVector { len: la, elem: ea },
            Type::Array {
                shape: sb,
                elem: eb,
            },
        ) if sb.len() == 2 => {
            if matches!((*la, sb[0]), (Dim::Static(k1), Dim::Static(k2)) if k1 != k2) {
                return Type::Failed(
                    "row-vector–matrix product: the row's length must match the matrix's \
                     leading dimension (spec §07)"
                        .into(),
                );
            }
            match promote2(Some(ea.as_ref()), Some(eb.as_ref())) {
                Type::Deferred => Type::Deferred,
                elem => Type::TVector {
                    len: sb[1],
                    elem: Box::new(elem),
                },
            }
        }
        _ => Type::Deferred,
    }
}

/// If `t` is forbidden as an array / table-column element (spec §03: arrays
/// hold scalars, strings, or arrays; §02: measures, likelihoods, functions, and
/// tuples may not appear inside arrays/records/tables), name the kind for a
/// diagnostic. `Any` (strings, holes), `Deferred`, and `Var` pass — they are
/// not yet known to be objects.
fn forbidden_array_element(t: &Type) -> Option<&'static str> {
    match t {
        Type::Record(_) => Some("a record"),
        Type::Tuple(_) => Some("a tuple"),
        Type::Table { .. } => Some("a table"),
        Type::Measure { .. } => Some("a measure"),
        Type::Kernel { .. } => Some("a kernel"),
        Type::Function { .. } => Some("a function"),
        Type::Likelihood { .. } => Some("a likelihood"),
        Type::Module => Some("a module"),
        _ => None,
    }
}

/// Name a CONCRETE non-measure type for the `superpose` argument-kind
/// diagnostic. `Deferred`/`Any`/`Var`/`Failed` are not-yet-known, not
/// known-wrong, so they pass (`None`) rather than being misreported.
fn non_measure_kind(t: &Type) -> Option<&'static str> {
    match t {
        Type::Measure { .. } | Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_) => None,
        Type::Scalar(_) => Some("a scalar"),
        Type::Array { .. } | Type::TVector { .. } => Some("an array"),
        Type::Record(_) => Some("a record"),
        Type::Tuple(_) => Some("a tuple"),
        Type::Table { .. } => Some("a table"),
        Type::Kernel { .. } => Some("a kernel"),
        Type::Function { .. } => Some("a function"),
        Type::Likelihood { .. } => Some("a likelihood"),
        Type::RngState => Some("an rng state"),
        Type::Module => Some("a module"),
    }
}

/// `vector(e1, …, en)` — a static-length array of the unified element type.
fn vector_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Type {
    // §03: array elements must be scalars, strings, or arrays — never records,
    // tables, tuples, or non-value objects. Rejecting these here is also what
    // keeps a vector-of-records from masquerading as a table-valued column
    // (both would otherwise store a `Record` column element; see `table_type`).
    let mut bad = false;
    for (node, t, _) in args {
        if let Some(kind) = forbidden_array_element(t) {
            inf.diags.push(crate::Diagnostic::error_at(
                *node,
                format!(
                    "array elements must be scalars, strings, or arrays (spec §03); got {kind}"
                ),
            ));
            bad = true;
        }
    }
    if bad {
        return Type::Failed("array element is not a scalar, string, or array".into());
    }
    let mut elem: Option<Type> = None;
    for (_, t, _) in args {
        elem = Some(match elem {
            None => t.clone(),
            Some(prev) if &prev == t => prev,
            Some(prev) => match promote2(Some(&prev), Some(t)) {
                Type::Deferred => Type::Any, // heterogeneous non-numeric
                p => p,
            },
        });
    }
    Type::Array {
        shape: Box::new([Dim::Static(args.len() as u32)]),
        elem: Box::new(elem.unwrap_or(Type::Any)),
    }
}

/// `table(col1 = v1, col2 = v2, …)` (spec §03 "Tables"): named equal-length
/// columns → a table. FlatPIR stores each column's per-row ELEMENT type (not the
/// column itself) plus a single shared `nrows` (§11 `(%table (%columns (name elem)
/// …) (%nrows N))`), so the leading dim is lifted out of the columns into `nrows`,
/// taken from the first column (the spec requires all columns equal-length).
///
/// A column is a **vector** or a **table** (spec §03). A vector column's element
/// may itself be an array (a 3-vector per row), kept verbatim. A **table-valued**
/// column contributes a record per row — one row of the sub-table — so its stored
/// element is `Record(sub-columns)` and its length is the sub-table's `nrows`
/// (`get(t, i)` then yields a row whose entry for that column is a record).
/// A non-vector / non-table (or `%deferred`) column leaves the table `%deferred`
/// — no valid table type can be formed (honesty over coverage). The `table(r)`
/// record-of-vectors form (spec §03) is not handled here and defers. `nrows` is
/// `%dynamic` when the first column's length is dynamic.
fn table_type(inf: &mut Inferencer<'_, '_>, named: &[NamedInfo]) -> Type {
    if named.is_empty() {
        return Type::Deferred;
    }
    let cols: Vec<(Symbol, &Type, NodeId)> =
        named.iter().map(|(n, node, t, _)| (*n, t, *node)).collect();
    build_table(inf, &cols)
}

/// Build a `%table` from `(column name, column-value type, anchor node)` triples
/// — shared by `table(col = …)` (named columns) and `table(r)` (record-of-vectors
/// auto-splat). Each column is a **vector** (store its element) or a **table**
/// (store its per-row record). The shared `nrows` is the first statically-known
/// column length; a later column whose static length differs is an equal-length
/// error (spec §03), anchored on that column's node.
fn build_table(inf: &mut Inferencer<'_, '_>, cols: &[(Symbol, &Type, NodeId)]) -> Type {
    if cols.is_empty() {
        return Type::Deferred;
    }
    let mut columns = Vec::with_capacity(cols.len());
    let mut nrows = Dim::Dynamic;
    for &(name, t, node) in cols {
        let (len, elem) = match t {
            Type::Array { shape, elem } if shape.len() == 1 => (shape[0], (**elem).clone()),
            Type::Table {
                columns: sub,
                nrows: sub_nrows,
            } => (*sub_nrows, Type::Record(sub.clone())),
            _ => return Type::Deferred,
        };
        match (nrows, len) {
            (Dim::Dynamic, _) => nrows = len,
            (Dim::Static(have), Dim::Static(got)) if have != got => {
                let col = inf.module.resolve(name).to_string();
                inf.diags.push(crate::Diagnostic::error_at(
                    node,
                    format!(
                        "table columns must have equal length (spec §03): column `{col}` has \
                         {got} rows, but an earlier column has {have}"
                    ),
                ));
            }
            _ => {}
        }
        columns.push((name, elem));
    }
    Type::Table {
        columns: columns.into(),
        nrows,
    }
}

/// `record(r)` on an argument already a record, or `table(t)` on an argument
/// already a table: §03/§04 auto-splatting is defined for the OTHER aggregate
/// kind (`record(t)` reads a table's columns, `table(r)` reads a record's
/// fields) and no spec sentence gives a same-kind call any meaning.
///
/// Before this, `record(record(a = 1.0, b = 2.0))` silently returned an EMPTY
/// `(%record )` (the constructor's `_` arm reads `named`, which is empty for a
/// positional call) and `table(t)` on a table silently returned `%deferred`
/// (`table_type` also reads `named`) — both a wrong type with no diagnostic.
///
/// flatppl-js treats this call as identity pass-through, an engine-leniency
/// rider flagging that the spec ruling is open (`TODO-flatppl-js.md`); this
/// does NOT invent that identity semantics, since the ruling could go either
/// way. It refuses with a location diagnostic instead, pending the ruling
/// (`TODO-flatppl-rust.md`).
/// True iff `t` is an array or a transposed vector — the shapes §07's comparison
/// rows do not admit. A `Table`/`Record` operand is NOT included: it takes the §04
/// auto-splat path, which is a separate rule and a separate diagnostic.
fn is_array_like(t: &Type) -> bool {
    matches!(t, Type::Array { .. } | Type::TVector { .. })
}

/// A short phrase naming an [`is_array_like`] shape, for
/// [`refuse_array_comparison`]'s message. `Type` has no `Display`, and the rank is
/// the part of the shape that makes the refusal legible.
fn array_shape_phrase(t: &Type) -> String {
    match t {
        Type::Array { shape, .. } => format!("a rank-{} array", shape.len()),
        Type::TVector { .. } => "a transposed vector".to_string(),
        _ => "a non-scalar".to_string(),
    }
}

/// An array operand to `equal`/`unequal`/`lt`/`le`/`gt`/`ge`.
///
/// §07 "Operator-equivalent functions" gives the comparison rows the domains
/// `reals` (`lt`, `le`, `gt`, `ge`) and "`integers`, `booleans`, strings"
/// (`equal`, `unequal`) — all SCALAR value-sets, where the `add`/`sub` rows in the
/// SAME table read "scalars or arrays of same shape". The contrast is deliberate and
/// visible side by side, so a comparison has no array domain to lower. §05 "Excluded
/// constructs" states the general rule from the other direction — "**No implicit
/// operator broadcasting.**" — and names `broadcast` and the dotted operators as the
/// elementwise route. (§05's bullet enumerates only the arithmetic operators, so §07's
/// Domains column is the load-bearing citation here, not that bullet's examples.)
///
/// This was accepted before, and the leniency was not harmless: `infer` typed
/// `gt(v, 3.0)` over a length-3 `v` as `Scalar(Boolean)` — a scalar — while the
/// StableHLO emitter broadcast the same node and emitted
/// `func.func @logdensity(%arg0: tensor<3xf32>) -> tensor<3xi1>`. The declared type
/// and the emitted ABI disagreed on the result's SHAPE, with no diagnostic on either
/// side. Refusing statically closes both halves at once.
///
/// The diagnostic names the dotted form, which is the working route: `gt.(v, 3.0)`,
/// `v .> 3.0` and `broadcast(gt, v, 3.0)` all reach [`broadcast_type`]'s elementwise
/// arm and type a boolean array of `v`'s shape.
fn refuse_array_comparison(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    let offender = args
        .iter()
        .map(|(_, t, _)| t)
        .chain(named.iter().map(|(_, _, t, _)| t))
        .find(|t| is_array_like(t))
        .cloned()
        .unwrap_or(Type::Deferred);
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!(
            "`{name}` expects scalar operands, got {}: spec §07 \"Operator-equivalent \
             functions\" gives the comparisons the scalar domains `reals` and \
             `integers`/`booleans`/strings, where `add`/`sub` in the same table read \
             \"scalars or arrays of same shape\", and §05 \"Excluded constructs\" states \
             \"No implicit operator broadcasting\". Apply it elementwise instead — \
             `{name}.(a, b)`, or the dotted operator — which gives a boolean array of the \
             operand's shape",
            array_shape_phrase(&offender)
        ),
    ));
    Type::Failed(format!("`{name}` applied to an array operand").into())
}

fn refuse_same_kind_constructor(inf: &mut Inferencer<'_, '_>, id: NodeId, name: &str) -> Type {
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!(
            "`{name}`'s sole positional argument is already a {name}; spec §03/§04 \
             auto-splatting converts the OTHER aggregate kind into a {name}, never a \
             {name} into itself — this call has no defined meaning and is refused \
             pending a spec ruling on same-kind construction"
        ),
    ));
    Type::Failed(format!("`{name}` applied to an argument already of that kind").into())
}

/// `record(t)`: a table's columns as a record of column vectors (spec §03, the
/// inverse of `table(r)`). A vector column becomes a length-`nrows` vector; a
/// table-valued column (stored as its per-row record) becomes the sub-table —
/// mirroring `get`-by-column access.
fn record_from_table(columns: &[(Symbol, Type)], nrows: Dim) -> Type {
    Type::Record(
        columns
            .iter()
            .map(|(n, elem)| {
                let col = match elem {
                    Type::Record(sub) => Type::Table {
                        columns: sub.clone(),
                        nrows,
                    },
                    e => Type::Array {
                        shape: Box::new([nrows]),
                        elem: Box::new(e.clone()),
                    },
                };
                (*n, col)
            })
            .collect(),
    )
}

/// `rowstack([rows…])`: an array of equal-length vectors becomes a matrix.
fn rowstack_type(a: Option<&Type>) -> Type {
    match a {
        Some(Type::Array { shape, elem }) if shape.len() == 1 => match elem.as_ref() {
            Type::Array {
                shape: inner,
                elem: cell,
            } if inner.len() == 1 => Type::Array {
                shape: Box::new([shape[0], inner[0]]),
                elem: cell.clone(),
            },
            _ => Type::Deferred,
        },
        _ => Type::Deferred,
    }
}

/// `sum`/`prod` over an array reduce to the element type.
/// The scalar a §07 reduction produces over elements of scalar type `elem`. The one
/// place that answer is written down, so the array form ([`reduce_type`]) and the
/// table form ([`table_reduction_type`]) cannot drift — and, more to the point,
/// cannot "agree" by sharing a mistake.
///
/// - `sum`/`prod` over BOOLEANS — `Integer`, by §03 "Bool": "In arithmetic contexts,
///   `false` is promoted to zero and `true` to one, permitting expressions such as
///   `true + true`, `3 * false`, and `sum(mask)` to count true entries". Zero and one
///   are §03 "Scalar types" `Integer`s, and §03 "Scalar value categories and sets"
///   fixes `booleans` $\subset$ `integers` $\subset$ `reals`, so `integers` is the
///   narrowest set the promotion lands in — a count, not a boolean. Keeping the
///   element type here made `sum([true, true, false])` a boolean, which the
///   StableHLO emitter honoured with a 1-bit `stablehlo.add`: parity, not a count.
/// - `sum`/`prod` otherwise — the element type. A sum of integers is an integer.
/// - `maximum`/`minimum` — the element type: they return an ELEMENT of the input
///   rather than a computed aggregate, matching their catalogue row's
///   `ElemScalarKind` result (an integer array's max is an integer). Booleans are
///   NOT promoted: $\max_i x_i$ selects an element and performs no arithmetic, so
///   §03's promotion sentence does not reach it, and a boolean array's max is a
///   boolean.
/// - `mean` — §07 defines it as $\bar{x} = \frac{1}{n}\sum_i x_i$, and the mean of
///   `[1, 2]` is `1.5`, so an INTEGER input gives a REAL. Complex stays complex
///   (§07's domain for `mean` is "real/complex arrays"). This is arithmetic, so it
///   outranks both the previous code and any convenience of keeping the element type.
/// - `var`/`std` — real, matching their catalogue rows and their "real arrays" domain.
/// - `lany`/`lall` — `Boolean`, whatever the column's element kind. §07 "Boolean
///   reductions" gives both the domain "boolean arrays" and describes each as a
///   truth value ("`true` if at least one element of `xs` is `true`"), and
///   `lor`/`land` are closed on `booleans`. Nothing is promoted: unlike `sum`, a
///   disjunction is not arithmetic, so §03 "Bool"'s promotion sentence does not
///   reach it.
/// - `median` — `Real`, matching its catalogue row and for the reason recorded
///   there: §07 averages two order statistics at even $n$, so `median([1, 2])` is
///   `1.5`.
fn reduced_scalar(head: &str, elem: ScalarType) -> ScalarType {
    match (head, elem) {
        ("sum" | "prod", ScalarType::Boolean) => ScalarType::Integer,
        ("sum" | "prod" | "maximum" | "minimum", e) => e,
        ("lany" | "lall", _) => ScalarType::Boolean,
        ("mean", ScalarType::Complex) => ScalarType::Complex,
        _ => ScalarType::Real,
    }
}

/// The element kind of an `aggregate(f_reduction, output_axes, body)` result when
/// `f_reduction` fixes it REGARDLESS of the body's element kind — `None` when the
/// body's kind is the answer.
///
/// §04 "Multi-axis aggregation" makes the result "an array of the shape declared by
/// `output_axes`" whose entries are `f_reduction` applied to the contracted slice.
/// So the entry type is whatever that reduction gives, which for three of §04's ten
/// eligible built-ins is not the body's own kind:
///
/// - `median` — real even over an integer body, for [`reduced_scalar`]'s reason.
/// - `lany`/`lall` — boolean whatever the body was; §07 "Boolean reductions" gives
///   both a truth value.
///
/// `mean`, `var` and `std` have the SAME mismatch (all three are real-valued over an
/// integer body) and are deliberately not listed: fixing them changes an
/// already-shipped type on a construct outside this batch. Recorded in
/// `flatppl-dev/TODO-flatppl-rust.md`, alongside the `sum` divergence
/// `stablehlo::Emitter::reduce_trailing_axes` documents from the other side. Every
/// axis-indexed body types `%deferred`, so the fallback lands on `Real` and the
/// three are right on the common case; only a genuinely integer-typed body
/// (`indicesof(...)`) surfaces it.
fn aggregate_result_kind(inf: &Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<ScalarType> {
    let Node::Const(op) = inf.module.node(args.first()?.0) else {
        return None;
    };
    match inf.module.resolve(*op) {
        "median" => Some(ScalarType::Real),
        "lany" | "lall" => Some(ScalarType::Boolean),
        _ => None,
    }
}

/// `sum`/`prod`/`mean` over an ARRAY (spec §07 Reductions). A scalar element type is
/// mapped by [`reduced_scalar`]; a non-scalar element (an array-of-arrays) keeps the
/// element type as before, since §07 does not pin down what reducing along one axis
/// of a nested array yields and this is not the place to guess.
/// True iff `a` is a rank-1-or-higher array (or `TVector`) whose scalar element kind is
/// `Boolean` — the guard on the `cumsum`/`cumprod` promotion arm. Nested elements are
/// not drilled: §07 gives the cumulative pair the domain "vectors", so a nested element
/// is out of domain and keeps the catalogue row rather than being promoted here.
fn bool_elem_array(a: Option<&Type>) -> bool {
    matches!(
        a,
        Some(Type::Array { elem, .. } | Type::TVector { elem, .. })
            if matches!(elem.as_ref(), Type::Scalar(ScalarType::Boolean))
    )
}

/// The §03-promoted result type of `cumsum`/`cumprod` over the boolean array `a`:
/// the argument's own shape AND orientation (§07 makes the cumulative pair
/// shape-preserving) with `Integer` elements. Only the element kind is promoted.
///
/// A `TVector` stays a `TVector`. §03 "Arrays" keeps a transposed vector a distinct
/// type, and §07 "Linear algebra" makes "the product of a transposed vector and a
/// matrix … a transposed vector" — so collapsing the orientation to a rank-1 `Array`
/// loses a type the spec pins: `cumsum(transpose(b)) * M` typed `%deferred` instead of
/// `%tvector`, since `mul_type` has no rule for a bare array against a matrix.
///
/// The one place this answer is written down, so the type arm and the value-set arm in
/// [`call_valueset`] cannot drift — the value-set is `ValueSet::natural_of` of exactly
/// this type, and it handles `TVector` too (`CartPow(elem, len)`, the same set an
/// `Array` of that length gives). Only ever called behind [`bool_elem_array`], so the
/// shapeless fallback is unreachable.
fn cumulative_bool_type(a: Option<&Type>) -> Type {
    if let Some(Type::TVector { len, .. }) = a {
        return Type::TVector {
            len: *len,
            elem: Box::new(Type::Scalar(ScalarType::Integer)),
        };
    }
    let shape: Box<[Dim]> = match a {
        Some(Type::Array { shape, .. }) => shape.clone(),
        _ => Box::new([Dim::Dynamic]),
    };
    Type::Array {
        shape,
        elem: Box::new(Type::Scalar(ScalarType::Integer)),
    }
}

fn reduce_type(head: &str, a: Option<&Type>) -> Type {
    match a {
        Some(Type::Array { elem, .. }) => match elem.as_ref() {
            Type::Scalar(s) => Type::Scalar(reduced_scalar(head, *s)),
            other => other.clone(),
        },
        Some(Type::Any) => Type::Any,
        _ => Type::Deferred,
    }
}

/// The result of a §07 **Table reductions** call: "When `sum`, `mean`, `var`,
/// `std`, `prod`, `maximum`, `minimum`, `median`, `lany`, or `lall` is applied to a
/// table, the reduction operates column-wise and returns a record whose fields are
/// the column names and values are the per-column reductions." So the result is a
/// `Record` with one field per column, named for the column.
///
/// `std` is in the set by the owner ruling of 2026-08-10 which adds it to that
/// paragraph (flatppl-design `4c93237`). **That commit is NOT on design `main`** — it
/// sits on the `mul-divide-rows` branch — so `std`'s membership rests on unmerged
/// spec, exactly as its splat exemption does. `prod`, `maximum`, `minimum` are in
/// the set by design PR #79 (owner-merge pending as of this change), which extends
/// the same paragraph to those three; the engine work lands ahead of the spec merge
/// per the owner's ruling, with the spec gap recorded in `flatppl-dev`.
/// `median`, `lany` and `lall` are in the set by the `missing-reductions` spec draft
/// (flatppl-design `ee4c6fb`), quoted above; that branch is likewise unmerged, and
/// this engine work lands ahead of it under the same ruling.
///
/// The per-column value is whatever that reduction gives for an array of the
/// column's element type, so the two forms agree by construction rather than by a
/// second set of rules:
///
/// - `sum`/`mean`/`prod`/`maximum`/`minimum` — the column's own element type,
///   mirroring [`reduce_type`]'s array arm (so a complex column sums to complex)
///   and, for `maximum`/`minimum`, the catalogue row's `ElemScalarKind` result.
/// - `var`/`std`/`median` — `Scalar(Real)`, mirroring their catalogue row's declared
///   `result: Scalar(Real)`, which is what they give for an array of any element
///   type.
/// - `lany`/`lall` — `Scalar(Boolean)`, likewise from their catalogue row.
///
/// §07 also states "Every column must support the reduction operation", which this
/// does NOT check: `median` over a boolean column and `lany` over a real column both
/// type without complaint. The gap is pre-existing for `maximum`/`minimum` over a
/// boolean column, the three new heads inherit it, and closing it needs a per-head
/// domain table §07 does not spell out — recorded in `flatppl-dev/TODO-flatppl-rust.md`.
///
/// A column whose per-row type is NOT a scalar (a vector-valued column) leaves the
/// whole call `%deferred`. §07 says only "Every column must support the reduction
/// operation" and does not say what reducing a column of vectors yields, so this
/// declines to invent one — `%deferred` is the honest no-rule-yet answer, and it is
/// what the call typed before this rule existed.
/// The value-set companion of [`table_reduction_type`]: a `cartprod(col = …)` record
/// set whose fields match the result record's, so the type and the set describe the
/// same value. Per-field set is the natural extent of that field's type, except
/// `var`/`std`, which keep the `nonnegreals` their catalogue row declares — a
/// variance is non-negative per column exactly as it is for an array.
///
/// `Unknown` whenever [`table_reduction_type`] declines to produce a record, so the
/// two never disagree about whether a record is being described.
fn table_reduction_valueset(head: &str, a: Option<&Type>) -> ValueSet {
    let Type::Record(fields) = table_reduction_type(head, a) else {
        return ValueSet::Unknown;
    };
    ValueSet::RecordSet(
        fields
            .iter()
            .map(|(name, ty)| {
                let set = match head {
                    "var" | "std" => ValueSet::NonNegReals,
                    _ => ValueSet::natural_of(ty),
                };
                (*name, set)
            })
            .collect(),
    )
}

fn table_reduction_type(head: &str, a: Option<&Type>) -> Type {
    let Some(Type::Table { columns, .. }) = a else {
        return Type::Deferred;
    };
    if !columns.iter().all(|(_, t)| matches!(t, Type::Scalar(_))) {
        return Type::Deferred;
    }
    // Shares `reduced_scalar` with the array form, so the two agree on a rule that
    // was checked against §07's formulas rather than on whatever each happened to do.
    let per_column = |col: &Type| match col {
        Type::Scalar(s) => Type::Scalar(reduced_scalar(head, *s)),
        other => other.clone(),
    };
    Type::Record(
        columns
            .iter()
            .map(|(name, col)| (*name, per_column(col)))
            .collect(),
    )
}

/// `get` with static selectors: integer indices consume array axes / pick
/// tuple components; string keys pick record fields. Anything dynamic or
/// sliced (`all` / `only` / axes) is deferred until the shape work.
fn get_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo], base: i64) -> Type {
    let Some((_, container, _)) = args.first() else {
        return Type::Deferred;
    };
    let mut current = container.clone();
    for (node, sel_ty, _) in &args[1..] {
        let selector = inf.module.node(*node).clone();
        current = match (&current, &selector) {
            (Type::Tuple(comps), Node::Lit(Scalar::Int(k))) => {
                match usize::try_from(k - base).ok().and_then(|i| comps.get(i)) {
                    Some(t) => t.clone(),
                    None => return Type::Failed("tuple index out of range".into()),
                }
            }
            (Type::Array { shape, elem }, Node::Lit(Scalar::Int(_))) => {
                if shape.len() == 1 {
                    elem.as_ref().clone()
                } else {
                    Type::Array {
                        shape: shape[1..].into(),
                        elem: elem.clone(),
                    }
                }
            }
            (Type::TVector { elem, .. }, Node::Lit(Scalar::Int(_))) => elem.as_ref().clone(),
            (Type::Record(fields), Node::Lit(Scalar::Str(s))) => {
                let sym = fields.iter().find(|(n, _)| inf.module.resolve(*n) == &**s);
                match sym {
                    Some((_, t)) => t.clone(),
                    None => return Type::Failed(format!("record has no field `{s}`").into()),
                }
            }
            // A table indexed by an integer is ROW access → the row record
            // (spec §03 "Each row of a table is a record"); a table-valued
            // column makes that entry a nested record. The columns already
            // store per-row element types, so the row record IS `Record(cols)`.
            // The index value is not needed for typing (no bounds check, as for
            // arrays).
            (Type::Table { columns, .. }, Node::Lit(Scalar::Int(_))) => {
                Type::Record(columns.clone())
            }
            // A table indexed by a column name is COLUMN access → the column as
            // a vector (spec §03); a table-valued column (stored as its per-row
            // record) returns the sub-table itself, not a vector of records.
            (Type::Table { columns, nrows }, Node::Lit(Scalar::Str(s))) => {
                match columns.iter().find(|(n, _)| inf.module.resolve(*n) == &**s) {
                    Some((_, Type::Record(sub))) => Type::Table {
                        columns: sub.clone(),
                        nrows: *nrows,
                    },
                    Some((_, colty)) => Type::Array {
                        shape: Box::new([*nrows]),
                        elem: Box::new(colty.clone()),
                    },
                    None => return Type::Failed(format!("table has no column `{s}`").into()),
                }
            }
            (Type::Any | Type::Deferred, _) => return current.clone(),
            // A non-literal selector: fall back to its inferred TYPE. Indexing
            // an array by an integer ARRAY is a GATHER (`a[idxs]` — result has
            // the index's shape and the container's element, spec §07 "array of
            // indices subset selection"); a scalar-integer selector consumes
            // the leading axis like a literal `Int`.
            _ => match (&current, sel_ty) {
                (
                    Type::Array { elem, .. },
                    Type::Array {
                        shape: ish,
                        elem: ie,
                    },
                ) if matches!(ie.as_ref(), Type::Scalar(ScalarType::Integer)) => Type::Array {
                    shape: ish.clone(),
                    elem: elem.clone(),
                },
                (Type::Array { shape, elem }, Type::Scalar(ScalarType::Integer)) => {
                    if shape.len() == 1 {
                        elem.as_ref().clone()
                    } else {
                        Type::Array {
                            shape: shape[1..].into(),
                            elem: elem.clone(),
                        }
                    }
                }
                _ => return Type::Deferred,
            },
        };
    }
    current
}

/// A single-axis power of a RECORD set is the set of tables, so its member is
/// a `Table`, not an array-of-records: spec §07 `load_data` states it directly
/// ("`cartprod` a record, and a power of a record set a table"), and
/// `ValueSet::natural_of` already gives `Type::Table { columns, nrows }` and
/// `cartpow(recordset, n)` the identical value-set. Applied inside
/// [`set_element_type`]'s `cartpow` arm so every construct declaring the set
/// (`elementof`, `external`, `load_data`) agrees on the member's type — and so
/// column access (`data.y`) resolves, which it cannot through an array element.
/// A multi-axis power of a record set (`cartpow(recordset, [2, 3])`) keeps its
/// array form: §03 gives no table with two row axes, and folding one axis into
/// `nrows` would drop the other.
fn table_of_record_power(shape: Box<[Dim]>, elem: Type) -> Type {
    match elem {
        Type::Record(columns) if shape.len() == 1 => Type::Table {
            columns,
            nrows: shape[0],
        },
        elem => Type::Array {
            shape,
            elem: Box::new(elem),
        },
    }
}

/// `lawof(x)`'s result type, for each of the three argument shapes §04 admits.
///
/// The measure case is DERIVED, not asserted. §04 as amended by flatppl-design#73
/// (@ `9d9a91c`, pending owner review) gives one equation — "`lawof(m)` is
/// `lawof(draw(m))`, the law of a draw from `m`" — and composing the two rules
/// this module already has for its halves types it mechanically:
///
/// - `draw(m)` is m's DOMAIN (`measure_domain`, the `"draw"` arm), and
/// - `lawof(<value : T>)` is a measure over `T`,
///
/// so `lawof(m)` is a measure over m's domain — which, for the `%normalized`
/// measure the gate admits, IS m's own type. §04 states that consequence
/// separately ("A probability measure of fixed or parameterized phase is its own
/// law, so `lawof(m)` is equivalent to `m` and `lawof` is idempotent"), so the
/// derivation and the prose agree.
///
/// **Both phases give the same TYPE, which is why no phase split appears here.**
/// §04 distinguishes them semantically — a fixed/parameterized measure is its own
/// law, while a stochastic one yields "the marginal law of a draw from it: the
/// mixture ν(B) = ∫ κ(z, B) dP(z)". A mixture of measures over `D` is still a
/// measure over `D`, and still a probability measure, so the two cases differ in
/// what the determiniser must BUILD, not in what infer records.
///
/// The result's `%mass` is the ARGUMENT's mass, not `%normalized`
/// unconditionally: [`lawof_mass_gate`] only admits a measure whose mass is
/// `%normalized` (a theorem) or `%deferred` (not yet inferred, §11) — never a
/// settled non-normalized class. Propagating the admitted mass gives
/// `%normalized` on the proven path and `%deferred` on the unproven one; this
/// is the ONLY place that mass is decided — the mass-level rule table has no
/// `"lawof"` arm, deliberately (see the pointer there).
///
/// Design-PR #73's option C (owner ruling, decisions-log 2026-08-18) is the
/// no-laundering rider: an engine that admits a `%deferred`-mass argument is
/// ASSUMING normalization, not proving it, and "must leave the result's `%mass`
/// `%deferred` rather than record it as `%normalized`" — stamping `%normalized`
/// here would record an assumption as knowledge, which §11's "strongest
/// statically KNOWN class" slot definition forbids. Before this fix the result
/// was always stamped `%normalized`, laundering the assumption; `lawof(joint())`
/// (the zero-component `joint` — the one source of a genuinely `%deferred`-mass
/// measure reachable from source, per `product_mass`'s empty-list arm) is the
/// executed red case.
///
/// A KERNEL argument lifts pointwise — §04: "On a non-nullary kernel, `lawof`
/// lifts pointwise, as the uniform kernel extension does for measure-algebra
/// operations" — so the result is a kernel over the same inputs whose output
/// measure is a law, carrying that output measure's mass onward the same way.
/// Before this it wrapped, producing a measure whose DOMAIN was a kernel.
/// [`lawof_mass_gate`] gates the kernel case by the identical three rules
/// (2026-08-19, `lawof-kernel-mass-maths.md`: the pointwise lift composes the
/// whole measure clause, settled-class error included, onto each output
/// measure the kernel generates), so this branch only ever sees a kernel whose
/// mass is `%normalized` or `%deferred` — the same two values the measure
/// branch above sees.
///
/// Every other argument is a VALUE, and keeps the original behaviour: a measure
/// over that value's type, `%normalized` unconditionally — there is no mass
/// slot on a value to propagate, and the law of a value is a probability
/// measure by definition. That is the overwhelmingly common spelling
/// (`lawof(y)`, `lawof(record(y = y))`) and the corpus's only one.
///
/// A `Likelihood` argument falls in that last bucket and so still types as a
/// measure OVER a likelihood. #73 is silent on likelihoods — it defines `lawof`
/// for a measure and for a kernel — so nothing here invents semantics for it; the
/// open question is recorded in the wave report rather than guessed at.
fn lawof_type(arg: Option<&Type>) -> Type {
    match arg {
        // `lawof(m)` = `lawof(draw(m))`: a measure over m's domain, carrying
        // m's own (gate-admitted) mass onward rather than reasserting it.
        Some(Type::Measure { domain, mass }) => Type::Measure {
            domain: domain.clone(),
            mass: *mass,
        },
        // Pointwise lift over a kernel's output measure — same propagation.
        Some(Type::Kernel { inputs, mass }) => Type::Kernel {
            inputs: inputs.clone(),
            mass: *mass,
        },
        // A value: the law of that value.
        other => Type::Measure {
            domain: Box::new(other.cloned().unwrap_or(Type::Any)),
            mass: Mass::Normalized,
        },
    }
}

/// The mass class of the slice's first argument, when that argument is a MEASURE
/// whose normalization cannot be established, as the `%name` to quote in a
/// diagnostic — `None` when the argument is not a measure, or is one whose mass
/// the caller must accept. Callers whose measure argument sits at a fixed offset
/// (`rand_mass_gate`, arg 1) pass the sub-slice starting there, so "first" means
/// "first of what was handed in", not "first of the call's own arguments".
///
/// Shared by [`lawof_mass_gate`], [`draw_mass_gate`], and [`rand_mass_gate`] so
/// the three cannot drift: all three implement "reject unless proven
/// `%normalized`, or not yet inferred", and the three narrowings that phrase
/// encodes are argued once here.
///
/// - `%finite` IS rejected. The rule quantifies over the mass CLASS: this reads
///   the class the mass rules produced and does no arithmetic of its own, so a
///   `%finite` measure is refused however normalized it looks here. Proving
///   normalization is the MASS RULE's job, and where a proof exists the class it
///   yields is `%normalized` before this is ever consulted — see
///   [`superpose_is_provably_normalized`], which is why a mixture with
///   sum-to-one weights reaches this function as `%normalized` rather than
///   arguing its way past a `%finite` verdict.
/// - `%deferred` PASSES. §11 defines it as "not yet inferred" rather than a mass
///   verdict, so rejecting it would turn every gap in mass inference into a
///   user-facing error on a possibly well-formed model. Note what this makes the
///   rule: **reject unless proven `%normalized`, or not yet inferred** — NOT
///   "reject what is proven unnormalized". The difference is `%unknown`, which §11
///   defines as "unknown total mass" and which IS rejected, without anything having
///   been proven about it, because the question is whether normalization was
///   established, not whether non-normalization was.
/// - Only `Type::Measure` is inspected here. A `Type::Kernel` argument is each
///   caller's own question, deliberately NOT folded into this shared function:
///   [`lawof_mass_gate`] decides it (2026-08-19, `lawof-kernel-mass-maths.md`)
///   with its own kernel arm, by the identical three rules — but `draw`/`rand`
///   over a kernel remain a separate, undecided question; see
///   [`draw_mass_gate`] for why `draw` does not extend to one. Folding the
///   kernel case in here would silently extend it to `draw`/`rand` too.
fn unprovable_normalization(args: &[ArgInfo]) -> Option<&'static str> {
    let (_, arg_ty, _) = args.first()?;
    let Type::Measure { mass, .. } = arg_ty else {
        return None;
    };
    match mass {
        // Proven normalized, or not yet inferred — let it through.
        Mass::Normalized | Mass::Deferred => None,
        Mass::Null => Some("%null"),
        Mass::Finite => Some("%finite"),
        Mass::LocallyFinite => Some("%locallyfinite"),
        Mass::Unknown => Some("%unknown"),
    }
}

/// `draw(m)`'s total-mass gate: you cannot draw from a measure that is not a
/// probability measure, and this engine will not quietly normalize one for you.
///
/// **§04 states the rule normatively** (`docs/04-design.md`, reification): "`x ~ m`
/// (equivalent to `x = draw(m)`) introduces a stochastic node `x` by drawing a
/// variate from a normalized measure (i.e. a probability measure) `m`." A `draw`
/// from anything else is therefore already ill-formed; the owner ruling is only
/// that the engine says so instead of normalizing quietly, because implicit
/// normalization makes a model's meaning depend on a step the user never wrote.
///
/// flatppl-design#73 corroborates §04 when its equation is read right-to-left:
/// #73 gives `lawof(m)` = `lawof(draw(m))` and requires `lawof`'s argument to be
/// `%normalized`, so a draw from an unnormalized measure has no law.
///
/// What §04 does NOT settle is how much proving a checker attempts. It says
/// "normalized measure", not how hard to work at showing a measure is one. That
/// question is answered in two places, and the split matters: the MASS RULES
/// prove what they can (see [`superpose_is_provably_normalized`] for the
/// sum-to-one mixture) and this gate then quantifies over the resulting class
/// (see [`unprovable_normalization`]). The classification a checker is expected
/// to reach is the part that may still owe spec text.
///
/// The gap this closes was surfaced measurably: `draw(truncate(lawof(…), S))` used
/// to lower to the marginal density gated on `S` with **no normalizer** — the
/// correct density of an unnormalized restriction, silently presented as a law.
/// `normalize(...)` is the escape, and the diagnostic says so.
///
/// **A KERNEL argument is deliberately NOT gated, and not because it is safe.**
/// §06's "Uniform kernel extension" is what would license reading `draw(K)`
/// pointwise — "On a kernel, the operation applies to the output measure at each
/// input point" — but that paragraph scopes itself to measure-algebra operations
/// and closes with "This applies to all measure-to-measure operations except
/// `jointchain` and `kchain`". `draw` is measure-to-VALUE, so the sentence does not
/// reach it. §06 arguably FORBIDS the case outright: "Operations that map measures
/// to values, like `totalmass`, `densityof`, and `logdensityof`, require closed
/// measures (i.e. nullary kernels) as inputs" — an exemplary list, and `draw` maps
/// a measure to a value. A NULLARY kernel is a measure by that same paragraph's
/// identity ("identify measures with nullary kernels") and so is covered by the
/// measure arm. So the open question is whether `draw(<non-nullary kernel>)`
/// deserves a static error of its own, which is a rule about shapes rather than
/// about mass; today it types `%deferred` via `measure_domain` and is left alone.
fn draw_mass_gate(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let offending = unprovable_normalization(args)?;
    let (arg_node, _, _) = args.first()?;
    inf.diags.push(crate::Diagnostic::error_at(
        *arg_node,
        format!(
            "`draw` requires a probability measure, but this argument's total mass is \
             `{offending}`: there is no draw from an unnormalized measure. Wrap it in \
             `normalize(...)` to state that intent — `draw` never normalizes its \
             argument"
        ),
    ));
    Some(Type::Failed("draw from an unnormalized measure".into()))
}

/// `rand(rstate, m)`'s total-mass gate: the same rule as [`draw_mass_gate`]
/// (§07's `rand` draws from a normalized measure exactly as `draw` does — #73's
/// equation is agnostic to which spelling produced the draw), reused via
/// [`unprovable_normalization`] and adapted only for `rand`'s argument order:
/// the measure is arg 1, not arg 0 (arg 0 is the rngstate).
fn rand_mass_gate(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let measure_arg = args.get(1..)?;
    let offending = unprovable_normalization(measure_arg)?;
    let (arg_node, _, _) = measure_arg.first()?;
    inf.diags.push(crate::Diagnostic::error_at(
        *arg_node,
        format!(
            "`rand` requires a probability measure, but this argument's total mass is \
             `{offending}`: there is no draw from an unnormalized measure. Wrap it in \
             `normalize(...)` to state that intent — `rand` never normalizes its \
             argument"
        ),
    ));
    Some(Type::Failed("rand from an unnormalized measure".into()))
}

/// `lawof(m)`'s total-mass gate: a MEASURE argument must be `%normalized`.
///
/// Spec §04 as amended by flatppl-design#73 (@ `9d9a91c`, pending owner review):
/// "`lawof(m)` requires `m`'s `%mass` to be `%normalized` (see total-mass
/// classes); anything else is a static error, since an unnormalized measure is
/// not its own law and admits no such mixture. `lawof` never normalizes its
/// argument; `normalize(m)` states that intent." So the diagnostic names
/// `normalize` as the escape.
///
/// Returns `Some(Type::Failed(_))` when it rejects, `None` to let the ordinary
/// rule run. Deliberately conservative, and deliberately narrow in three ways:
///
/// - It rejects on the mass class the checker can PROVE is not `%normalized`,
///   which includes `%finite`. A `%finite` `superpose` whose weights happen to sum
///   to one is rejected: the class is what the rule quantifies over, not the
///   arithmetic, and a checker that tried to discharge the arithmetic would accept
///   some models and reject equivalent ones depending on how much constant folding
///   had run.
/// - `%deferred` PASSES. §04's "anything else" reads literally as including it,
///   but `%deferred` means "not yet inferred" rather than "not normalized" (§11),
///   so rejecting it would convert every gap in mass inference into a user-facing
///   error on a model that may be perfectly well-formed. So the rule this
///   implements is **reject unless proven `%normalized`, or not yet inferred** —
///   NOT "reject what is proven unnormalized". The difference is `%unknown`, which
///   §11 defines as "unknown total mass": it is rejected without anything having
///   been proven about it, because the gate's question is whether normalization was
///   established, not whether non-normalization was.
/// - A `Type::Kernel` argument is gated too, by the SAME three rules, pointwise
///   (`lawof-kernel-mass-maths.md`, ruled 2026-08-19): §04's "On a non-nullary
///   kernel, `lawof` lifts pointwise" composes the whole measure clause above —
///   requirement, settled-class error, and no-laundering rider alike — onto each
///   output measure the kernel generates, and §11 puts the kernel `%mass` slot
///   under the same "statically known" definition ("respectively all measures
///   generated by the kernel"). The maths is exhaustive: wherever `lawof(K)` is
///   defined at all, every output measure has mass 1 (the identity law at a
///   fixed/parameterized point, the marginal-mixture integral at a stochastic
///   one), so a settled non-`%normalized` kernel result never describes a value
///   — it types an expression that has none, the same defect the measure arm
///   exists to catch.
fn lawof_mass_gate(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    if let Some(offending) = unprovable_normalization(args) {
        let (arg_node, _, _) = args.first()?;
        inf.diags.push(crate::Diagnostic::error_at(
            *arg_node,
            format!(
                "`lawof` requires a `%normalized` measure (spec §04), but this argument's \
                 total mass is `{offending}`: an unnormalized measure is not its own law. \
                 Wrap it in `normalize(...)` to state that intent — `lawof` never \
                 normalizes its argument"
            ),
        ));
        return Some(Type::Failed("lawof of an unnormalized measure".into()));
    }
    let (arg_node, arg_ty, _) = args.first()?;
    let Type::Kernel { mass, .. } = arg_ty else {
        return None;
    };
    let offending = match mass {
        // Proven normalized at every input, or not yet inferred — let it
        // through, exactly as the measure arm does.
        Mass::Normalized | Mass::Deferred => return None,
        Mass::Null => "%null",
        Mass::Finite => "%finite",
        Mass::LocallyFinite => "%locallyfinite",
        Mass::Unknown => "%unknown",
    };
    inf.diags.push(crate::Diagnostic::error_at(
        *arg_node,
        format!(
            "`lawof` lifts pointwise over a kernel (spec §04), but this kernel's output \
             measures' total mass is `{offending}`: an unnormalized measure is not its \
             own law. Wrap it in `normalize(...)` to state that intent — `lawof` never \
             normalizes its argument"
        ),
    ));
    Some(Type::Failed("lawof of an unnormalized kernel".into()))
}

/// The element type of a set expression (`elementof` / `external` argument),
/// read structurally — sets are not first-class in the type grammar.
fn set_element_type(inf: &mut Inferencer<'_, '_>, node: Option<NodeId>) -> Type {
    let Some(node) = node else {
        return Type::Deferred;
    };
    let module = &*inf.module;
    match module.node(node) {
        Node::Const(sym) => match module.resolve(*sym) {
            "reals" | "posreals" | "nonnegreals" | "unitinterval" => Type::Scalar(ScalarType::Real),
            "integers" | "posintegers" | "nonnegintegers" => Type::Scalar(ScalarType::Integer),
            "booleans" => Type::Scalar(ScalarType::Boolean),
            "complexes" => Type::Scalar(ScalarType::Complex),
            "rngstates" => Type::RngState,
            "anything" => Type::Any,
            _ => Type::Deferred,
        },
        Node::Call(c) => match c.head {
            flatppl_core::CallHead::Builtin(op) => match module.resolve(op).to_string().as_str() {
                "interval" => Type::Scalar(ScalarType::Real),
                "cartpow" => {
                    // `cartpow(S, size)` where `size` is an integer (1-D) or a
                    // vector of positive integers (multi-axis), §03 "Cartesian
                    // power". `count_dims` reads a `vector` literal as one dim
                    // per element, so `cartpow(reals, [2, 3])` yields a rank-2
                    // (2×3) array — not the rank-1 dynamic a single-dim read
                    // would give (the legacy `cartpow(S, d1, d2, …)` arity is
                    // not in the spec; only arg 1 is the size).
                    let (set_arg, size_arg) = (c.args.first().copied(), c.args.get(1).copied());
                    // The size is required (§03 "Cartesian power"); a missing
                    // size is ill-formed — reject it here too, consistent with
                    // the `cartpow` type arm and `set_call_valueset` (an omitted
                    // size is not a dynamic size; a dynamic size is written
                    // `cartpow(S, n)` with a non-literal `n`).
                    match size_arg {
                        None => Type::Failed("cartpow expects (set, size)".into()),
                        Some(size_arg) => {
                            let elem = set_element_type(inf, set_arg);
                            let shape = count_dims(inf, size_arg);
                            table_of_record_power(shape, elem)
                        }
                    }
                }
                "stdsimplex" => {
                    // `stdsimplex(n)` is the (n-1)-simplex {x ∈ ℝⁿ : xᵢ ≥ 0,
                    // Σxᵢ = 1} embedded in ℝⁿ (§03 "Standard simplex"): an
                    // element is a length-n real vector. The ≥0 / sum-to-1
                    // constraint lives in the value-set slot (`StdSimplex`), not
                    // the scalar type — so the element type is a rank-1 real
                    // array, mirroring `cartpow(reals, n)`.
                    let size_arg = c.args.first().copied();
                    let dim = size_arg.map_or(Dim::Dynamic, |n| resolve_dim(inf, n));
                    Type::Array {
                        shape: Box::new([dim]),
                        elem: Box::new(Type::Scalar(ScalarType::Real)),
                    }
                }
                "cartprod" => {
                    // Clone to release the module borrow before recursing into
                    // set_element_type (mirrors the pattern used in set_expr_valueset).
                    let c = c.clone();
                    if !c.named.is_empty() {
                        let fields: Vec<(Symbol, Type)> = c
                            .named
                            .iter()
                            .map(|na| (na.name, set_element_type(inf, Some(na.value))))
                            .collect();
                        if fields.iter().any(|(_, t)| matches!(t, Type::Deferred)) {
                            Type::Deferred
                        } else {
                            Type::Record(fields.into())
                        }
                    } else {
                        // Positional `cartprod` is a set of ARRAYS, not a tuple
                        // (spec §03): a member is the `cat` of one element per
                        // component, so the element type follows the same
                        // shape-class `cat` rule as a positional `joint` variate
                        // — all-scalar components → a length-n vector, all-vector
                        // components → a concatenated vector. The per-position
                        // membership lives in the value-set slot (`CartProd`),
                        // not the type. A mixed shape class (scalar with vector)
                        // defers, since §06/§07 `cat` forbid that concatenation.
                        let parts: Vec<Type> = c
                            .args
                            .iter()
                            .map(|&a| set_element_type(inf, Some(a)))
                            .collect();
                        // A member is the `cat` of one element per component
                        // (the same shape rule as `joint` variates); mixing
                        // shape classes is a static error (§03 cartprod mirrors
                        // §06 joint; §07 `cat` forbids scalar+vector).
                        cat_or_diagnose(inf, node, "cartprod", &parts)
                    }
                }
                _ => Type::Deferred,
            },
            _ => Type::Deferred,
        },
        _ => Type::Deferred,
    }
}

/// The domain of a measure type, for `draw` / `rand`.
fn measure_domain(m: Option<&Type>) -> Type {
    match m {
        Some(Type::Measure { domain, .. }) => domain.as_ref().clone(),
        _ => Type::Deferred,
    }
}

/// The field names a `disintegrate` selector picks (spec §06: works like `get` —
/// `"b"` selects field `b`; `["b", "c"]` selects `b` and `c`). `Some` only when
/// every entry is a literal string (a bare `Scalar::Str`, or a `vector(...)` of
/// `Scalar::Str`); index selectors and non-literals ⇒ `None` (caller defers).
fn selector_field_names(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<Vec<Box<str>>> {
    match inf.module.node(node).clone() {
        Node::Lit(Scalar::Str(s)) => Some(vec![s]),
        Node::Call(c)
            if matches!(c.head, CallHead::Builtin(op)
                if inf.module.resolve(op) == "vector") =>
        {
            let names: Option<Vec<Box<str>>> = c
                .args
                .iter()
                .map(|&a| match inf.module.node(a) {
                    Node::Lit(Scalar::Str(s)) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            // An empty selector (`[]`) lowers to `(vector)` with no args —
            // `disintegrate([], M)` has no defined meaning (a zero-field selected
            // subset is a vacuous disintegration). Return `None` so the caller
            // falls back to the deferred result rather than fabricating an output.
            names.filter(|v| !v.is_empty())
        }
        _ => None,
    }
}

/// `disintegrate(selector, joint)` (spec §06) → `(kernel, marginal)`. When the
/// joint is a record-domain measure and the selector statically names a non-empty
/// proper subset of its fields, the marginal is the record of the COMPLEMENT
/// (unselected) fields and the kernel's inputs are those complement variate names
/// (the conditioning variates). The kernel's OUTPUT domain (the selected
/// variates) is not carried by `Type::Kernel`, so it stays implicit. Falls back
/// to empty kernel inputs + a `%deferred` marginal domain when the joint isn't a
/// record measure or the selector isn't a static field-name set.
fn disintegrate_type(inf: &mut Inferencer<'_, '_>, call: &Call, args: &[ArgInfo]) -> Type {
    let part_mass = match arg_ty(args, 1) {
        Some(Type::Measure {
            mass: Mass::Normalized,
            ..
        }) => Mass::Normalized,
        _ => Mass::Unknown,
    };
    let selected = call
        .args
        .first()
        .and_then(|&n| selector_field_names(inf, n));
    let (inputs, marginal_domain): (Box<[Symbol]>, Type) = match (arg_ty(args, 1), selected) {
        (Some(Type::Measure { domain, .. }), Some(sel)) => match domain.as_ref() {
            Type::Record(fields) => {
                let is_sel = |n: &Symbol| sel.iter().any(|s| inf.module.resolve(*n) == &**s);
                let all_present = sel
                    .iter()
                    .all(|s| fields.iter().any(|(n, _)| inf.module.resolve(*n) == &**s));
                let complement: Vec<(Symbol, Type)> =
                    fields.iter().filter(|(n, _)| !is_sel(n)).cloned().collect();
                if all_present && !complement.is_empty() {
                    let inputs: Box<[Symbol]> = complement.iter().map(|(n, _)| *n).collect();
                    (inputs, Type::Record(complement.into()))
                } else {
                    (Box::new([]), Type::Deferred)
                }
            }
            _ => (Box::new([]), Type::Deferred),
        },
        _ => (Box::new([]), Type::Deferred),
    };
    Type::Tuple(Box::new([
        Type::Kernel {
            inputs,
            mass: part_mass,
        },
        Type::Measure {
            domain: Box::new(marginal_domain),
            mass: part_mass,
        },
    ]))
}

/// `iid(M, n)`: n iid draws bundle into an array over M's domain. A literal
/// count (or literal count vector) gives static dims; anything computed is
/// dynamic until fixed-value const-eval lands (engine-concepts §17.1).
///
/// A SCALAR count over a RECORD-valued `M` is the one shape §11 gives its own
/// form: `(%table (%columns …) (%nrows N))`, not an array of records — design
/// PR #83 (owner ruling, decisions-log 2026-08-18): "the text is correct (§11
/// gives `%table` its own form … ); rust types `%array` instead", now fixed.
/// §03's Cartesian power backs the reading too: "When `S` is a record set, the
/// power is the set of tables with those columns", with its own worked example
/// scalar (`cartpow(cartprod(a = reals, b = posreals), n)` is "the set of
/// `n`-row tables"). `count_dims` gives a scalar `n` exactly one dim
/// (`Box::new([..])`, both the literal-int and the dynamic-fallback arms), so
/// `shape.len() == 1` is precisely the scalar case — no multi-axis shape is
/// ever length 1, since a `vector`/`Vec` count contributes one dim per element.
///
/// A MULTI-axis count (`shape.len() != 1`, e.g. `iid(M, [2, 3])`) has NO table
/// reading — a table has one row axis — and stays array-of-records, untouched:
/// #83's own §03 tension notes "a multi-axis power of a record set has no
/// table reading at all". A non-record `M` is likewise untouched, falling to
/// the same array arm it always used.
fn iid_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Type {
    let domain = match arg_ty(args, 0) {
        Some(Type::Measure { domain, .. }) => domain.as_ref().clone(),
        _ => return Type::Deferred,
    };
    let Some((count_node, _, _)) = args.get(1) else {
        return Type::Deferred;
    };
    let shape = count_dims(inf, *count_node);
    let result_domain = match (&domain, shape.as_ref()) {
        (Type::Record(fields), [nrows]) => Type::Table {
            columns: fields.clone(),
            nrows: *nrows,
        },
        _ => Type::Array {
            shape,
            elem: Box::new(domain),
        },
    };
    Type::Measure {
        domain: Box::new(result_domain),
        mass: Mass::Deferred,
    }
}

/// A measure over a length-`len` trajectory in `init`'s state space (spec §06
/// `markovchain` / `kscan`): domain is `array[len]` of `init`'s type. The
/// initial state is excluded from the trajectory, so the element type is
/// exactly `init`'s. Record-state trajectories are tables (spec) — not built
/// here; a record `init` yields a deferred domain. Mass is left `Deferred` for
/// `fill_mass` to set from the kernel class.
fn trajectory_measure(init: Option<&Type>, len: Dim) -> Type {
    match init {
        Some(t @ (Type::Scalar(_) | Type::Array { .. })) => Type::Measure {
            domain: Box::new(Type::Array {
                shape: Box::new([len]),
                elem: Box::new(t.clone()),
            }),
            mass: Mass::Deferred,
        },
        _ => Type::Measure {
            domain: Box::new(Type::Deferred),
            mass: Mass::Deferred,
        },
    }
}

/// Scalar element kind of `t`, drilling array nesting; `None` for
/// non-scalar/non-array types.
fn elem_scalar_kind_of(t: &Type) -> Option<ScalarType> {
    match t {
        Type::Scalar(s) => Some(*s),
        Type::Array { elem, .. } => elem_scalar_kind_of(elem),
        _ => None,
    }
}

/// `t` with every array dim (at every nesting level) replaced by `%dynamic`,
/// preserving rank and element type. For ops that keep an argument's rank and
/// element but change its sizes (`tile`, `cat`).
fn with_dynamic_dims(t: &Type) -> Type {
    match t {
        Type::Array { shape, elem } => Type::Array {
            shape: vec![Dim::Dynamic; shape.len()].into_boxed_slice(),
            elem: Box::new(with_dynamic_dims(elem)),
        },
        other => other.clone(),
    }
}

/// The output axes of an `aggregate`/`metricsum` — the `(%axis …)` names in the
/// `output_axes` vector literal `[.i, .k]`, in order (one result axis each).
/// `None` when the axis list isn't a literal vector (rank not statically known).
fn output_axis_names(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<Vec<Symbol>> {
    let Node::Call(c) = inf.module.node(node) else {
        return None;
    };
    if !matches!(c.head, flatppl_core::CallHead::Builtin(op)
        if inf.module.resolve(op) == "vector")
    {
        return None;
    }
    let mut out = Vec::new();
    for &a in c.args.iter() {
        if let Node::Axis(ax) = inf.module.node(a) {
            out.push(ax.name);
        }
    }
    Some(out)
}

/// All dims of `t` flattened across array nesting (a nested-vector matrix
/// `Array[r]{ Array[c]{e} }` flattens to `[r, c]`), so an index position maps to
/// a single extent.
fn flatten_dims(t: &Type) -> Vec<Dim> {
    match t {
        Type::Array { shape, elem } => {
            let mut v = shape.to_vec();
            v.extend(flatten_dims(elem));
            v
        }
        _ => Vec::new(),
    }
}

/// Walk an `aggregate`/`metricsum` body collecting, for each axis name, the
/// input dim it indexes: an index `arr[…, ax_k, …]` (`get`/`get0`) binds `ax_k`
/// to `arr`'s flattened dim at that position. First binding wins (einsum
/// consistency); axes that never index a statically-shaped array stay absent
/// (→ dynamic).
fn collect_axis_dims(
    inf: &mut Inferencer<'_, '_>,
    node: NodeId,
    out: &mut std::collections::HashMap<Symbol, Dim>,
) {
    let Node::Call(c) = inf.module.node(node).clone() else {
        return;
    };
    if let flatppl_core::CallHead::Builtin(op) = c.head {
        let name = inf.module.resolve(op).to_string();
        if (name == "get" || name == "get0") && !c.args.is_empty() {
            let arr_ty = inf.infer_node(c.args[0]).0;
            let flat = flatten_dims(&arr_ty);
            for (k, &idx) in c.args.iter().enumerate().skip(1) {
                if let Node::Axis(ax) = inf.module.node(idx) {
                    if let Some(&d) = flat.get(k - 1) {
                        out.entry(ax.name).or_insert(d);
                    }
                }
            }
        }
    }
    for &a in c.args.iter() {
        collect_axis_dims(inf, a, out);
    }
}

/// The phase of `lawof(arg)` — the phase of the **reified law** of `arg`.
///
/// `lawof` absorbs the stochasticity of `draw` ancestors into the law, so the
/// result is deterministic: parameterized if the law depends on a free
/// `elementof` leaf, else fixed; never stochastic (spec §04 "Phase of the
/// reified law"). We re-derive the phase over the argument's ancestor subgraph
/// with two overrides vs the normal join: a `draw` contributes the law-phase of
/// its *measure operand* (absorbing the draw), and the recursion bottoms out at
/// `elementof` (parameterized) / fixed leaves — mirroring how `functionof`
/// traces to parametric leaves. Ref/`draw` cycles are bounded by `depth`.
fn law_phase(inf: &mut Inferencer<'_, '_>, node: NodeId, depth: u32) -> Phase {
    if depth > 64 {
        return Phase::Parameterized; // safe non-stochastic fallback
    }
    match inf.module.node(node).clone() {
        Node::Ref(r) if r.ns == flatppl_core::RefNs::SelfMod => {
            match inf.module.binding_by_name(r.name) {
                Some(b) => {
                    let rhs = inf.module.binding(b).rhs;
                    law_phase(inf, rhs, depth + 1)
                }
                None => Phase::Parameterized,
            }
        }
        Node::Call(c) => match c.head {
            flatppl_core::CallHead::Builtin(op) => match inf.module.resolve(op) {
                // Parametric leaf: the law depends on this free input.
                "elementof" => Phase::Parameterized,
                // Closed-over fixed leaves.
                "external" | "load_data" | "load_module" | "standard_module" => Phase::Fixed,
                // Absorb: the draw's stochasticity collapses into the law of
                // the measure it draws from.
                "draw" => c
                    .args
                    .first()
                    .map_or(Phase::Fixed, |&m| law_phase(inf, m, depth + 1)),
                // Any other node (measure constructor, container, arithmetic,
                // nested `lawof`, …): join the law-phases of its inputs.
                _ => c.args.iter().fold(Phase::Fixed, |acc, &a| {
                    join_phase(acc, law_phase(inf, a, depth + 1))
                }),
            },
            // User-callable application within a law: conservatively
            // parameterized (deterministic, may depend on inputs).
            _ => Phase::Parameterized,
        },
        // Literals and named constants are fixed; anything else (holes,
        // cross-module refs) is conservatively non-stochastic.
        Node::Lit(_) | Node::Const(_) => Phase::Fixed,
        _ => Phase::Parameterized,
    }
}

/// `joint(a = M1, b = M2, …)` — a measure over the record of the components'
/// domains (the positional form is deferred with the shape work).
///
/// A KERNEL component makes the whole `joint` a kernel — see
/// [`kernel_joint_type`].
fn joint_type(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    if args
        .iter()
        .any(|(_, t, _)| matches!(t, Type::Kernel { .. }))
        || named
            .iter()
            .any(|(_, _, t, _)| matches!(t, Type::Kernel { .. }))
    {
        return kernel_joint_type(inf, id, args, named);
    }
    // A mixed spelling is a static error in the measure arms too — see
    // [`refuse_mixed_joint_spelling`] and [`kernel_joint_type`]'s doc comment.
    // Before this, reading only `named` whenever it was non-empty silently
    // DROPPED every positional component: `joint(Normal(0.0, 1.0), b =
    // Exponential(1.0))` typed over `record{b}` alone.
    if !args.is_empty() && !named.is_empty() {
        return refuse_mixed_joint_spelling(inf, id);
    }
    // Keyword form `joint(a = M1, b = M2, …)`: a measure over a RECORD, each
    // component variate under its name (a record-valued component nests under
    // the name, not merged — spec §06).
    if !named.is_empty() {
        let mut fields = Vec::with_capacity(named.len());
        for (name, _, t, _) in named {
            match t {
                Type::Measure { domain, .. } => fields.push((*name, domain.as_ref().clone())),
                _ => return Type::Deferred,
            }
        }
        return Type::Measure {
            domain: Box::new(Type::Record(fields.into())),
            mass: Mass::Deferred,
        };
    }
    // Positional form `joint(M1, M2, …)`: the variate is the `cat` of the
    // component variates (spec §06) — the single `cat` shape rule (shared with
    // `cat` / positional `cartprod`); mixing shape classes is a static error.
    // (Not a record-per-component — that is the keyword form above.)
    let mut domains = Vec::with_capacity(args.len());
    for (_, t, _) in args {
        match t {
            Type::Measure { domain, .. } => domains.push(domain.as_ref().clone()),
            _ => return Type::Deferred,
        }
    }
    Type::Measure {
        domain: Box::new(cat_or_diagnose(inf, id, "joint", &domains)),
        mass: Mass::Deferred,
    }
}

/// A `joint` mixing positional and keyword components is a static error in
/// EVERY arm — measure, kernel, and (once inference reaches it) don't-know —
/// per §06 spelling two forms and no third: `joint(M1, M2, ...)` and
/// `joint(name1 = M1, name2 = M2, ...)`. `determinizer`'s `lower_joint`
/// already refuses the shape in the same words, so typing it here was only
/// ever a deferral of that refusal to a later pass.
fn refuse_mixed_joint_spelling(inf: &mut Inferencer<'_, '_>, id: NodeId) -> Type {
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        "`joint` mixes positional and keyword components: a `joint` is either \
         the positional cat-variate form or the keyword record-variate form, \
         not both (spec §06)",
    ));
    Type::Failed("joint mixes positional and keyword components".into())
}

/// `joint(K1, K2, …)` where at least one component is a KERNEL — the fan-out
/// kernel of spec §06's `joint` entry ("a kernel that fans a single input out to
/// all component kernels, so each of them receives the same input"). The five
/// semantic questions the sentence leaves open are settled in
/// `flatppl-dev/kernel-joint-q4-maths.md` and written into §06 by
/// flatppl-design#85:
///
/// - **Inputs: the union by name** (Q1). "The result's inputs are the union of
///   the component kernels' inputs by name; a component receives the inputs it
///   declares and is unaffected by the others." A name declared by several
///   components binds once and fans to each of them. First-occurrence order, so
///   the signature reads in source order rather than by symbol id.
/// - **Measure components are the nullary case** (Q3) — §06 "Uniform kernel
///   extension": "we unify measures and kernels and identify measures with
///   nullary kernels". They contribute nothing to the union, which is why an
///   all-measure `joint` never reaches here and a mixed one is legal.
/// - **The keyword form still names a record variate** (Q2), but §11's
///   `(%kernel (%inputs …) (%mass …))` carries no output domain, so the variate
///   has nowhere to land in `Type::Kernel`. It becomes visible on APPLICATION,
///   where [`kernel_joint_result_type`] builds the record (keyword) or `cat`
///   (positional).
/// - **RETAIN is a trace fact, not a type fact** (Q4). Whether the output law is
///   correlated or a product turns on node identity in the components' carried
///   traces, which no type slot records; the determiniser reads the traces.
/// - **Mass: the qualified product rule** (Q5), shared verbatim with the measure
///   case through [`joint_mass`]. At each input the output IS a measure-`joint`,
///   so the kernel case adds nothing of its own — §11: a kernel's `%mass` is
///   "the total-mass class of the output measure, uniform over all inputs".
///
/// A component that is neither a measure nor a kernel leaves the whole `joint`
/// `%deferred`, exactly as the measure arms do.
///
/// **A MIXED spelling — positional and keyword components in one call — is a
/// static error here.** §06 spells two forms and no third: `joint(M1, M2, ...)`
/// and `joint(name1 = M1, name2 = M2, ...)`. Reading only `named` whenever it is
/// non-empty, which is what the measure arms do, silently DROPS every positional
/// component, and for a kernel that means dropping its inputs from the union —
/// contradicting #85's own sentence, which unions "the component kernels' inputs"
/// with no qualification by spelling.
///
/// **The decision rests on §06 spelling two forms and no third**, which is the line
/// `determinizer`'s `lower_joint` already takes in the same words: "joint mixes
/// positional and keyword components; a joint is either the positional cat-variate
/// form or the keyword record-variate form, not both". So typing the shape was only
/// ever a deferral of that refusal to a later pass.
///
/// A second route — rewriting the call by §06's own relabel equivalence
/// (`joint(a = M1, b = M2)` "is equivalent to `joint(relabel(M1, ["a"]),
/// relabel(M2, ["b"]))`") and letting the shape-class rule judge the resulting
/// positional call — agrees only for a SCALAR positional component, and does not
/// generalise. The rule permits "all records with distinct field names", so a
/// record-variate positional component survives the reduction: for
/// `KR = kernelof(record(p = a1, q = a2), z = z)`, the rewrite
/// `joint(KR, relabel(K2, ["r"]))` is a joint of two records and types, while the
/// mixed `joint(KR, r = K2)` refuses here. Whatever §06 permits, this arm could not
/// apply the shape-class rule anyway: it needs each component's output variate,
/// which `Type::Kernel` does not carry (Q2).
///
/// The refusal itself lives in [`refuse_mixed_joint_spelling`], shared with
/// `joint_type`'s measure arms — the MEASURE arms dropped a positional
/// component the same way before that fix (`joint(Normal(0.0, 1.0), b =
/// Exponential(1.0))` typed over `record{b}` alone); this decision applies to
/// them unchanged, since nothing above turns on the component being a kernel.
fn kernel_joint_type(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    if !args.is_empty() && !named.is_empty() {
        return refuse_mixed_joint_spelling(inf, id);
    }
    let components: Vec<(NodeId, &Type)> = if named.is_empty() {
        args.iter().map(|(n, t, _)| (*n, t)).collect()
    } else {
        named.iter().map(|(_, n, t, _)| (*n, t)).collect()
    };
    let mut inputs: Vec<Symbol> = Vec::new();
    for (_, t) in &components {
        match t {
            Type::Kernel {
                inputs: component, ..
            } => {
                for name in component.iter() {
                    if !inputs.contains(name) {
                        inputs.push(*name);
                    }
                }
            }
            // Nullary: contributes no input (Q3).
            Type::Measure { .. } => {}
            _ => return Type::Deferred,
        }
    }
    if let Some(failed) = diagnose_shared_node_input_names(inf, id, &components) {
        return failed;
    }
    Type::Kernel {
        inputs: inputs.into(),
        mass: joint_mass(inf, args, named),
    }
}

/// What a `joint` component binds as boundary inputs, for the ancestry clause.
enum Binds {
    /// A LOCAL reification whose boundary this pass read, as `(input name, target
    /// name)` pairs.
    Declared(Vec<(Symbol, Symbol)>),
    /// Proven to bind nothing. §06 "Uniform kernel extension" identifies a measure
    /// with a nullary kernel, so a `Type::Measure` component declares no boundary
    /// input at all — the case `kernel-joint-w1-maths.md` calls out by name ("in
    /// particular a measure component, which binds nothing").
    ///
    /// A `Type::Kernel` component whose boundary this pass cannot read is NOT this:
    /// it is unknown, and it stays excluded from the check entirely.
    Nothing,
}

/// The Q1/W1 ancestry clause (§06 `joint` entry, flatppl-design#85):
/// "Components that share a stochastic node must agree on that node's ancestry:
/// every ancestor of the shared node that any component binds as a boundary input
/// must be bound by every sharing component, under the same input name. A `joint`
/// in which a sharing component binds such an ancestor under a different name, or
/// does not bind it at all — in particular a measure component, which binds
/// nothing — is a static error."
///
/// Why it is an error rather than a convention: under union-by-name the retained
/// node has one parent value. Two components substituting its boundary ancestor
/// with DIFFERENT inputs leave that parent undefined at any application where the
/// two inputs differ (`kernel-joint-q4-maths.md` §4), and a component that binds
/// the ancestor under NO name reads the ambient module parameter instead, so the
/// same single node would need two laws at once (`kernel-joint-w1-maths.md` §3).
/// Both halves are one incoherence in one proof shape, which is why they are one
/// clause and not two rules. The all-or-none boundary rule (§04 "Specifying
/// reification boundaries") already excludes the half-substituted and
/// same-name-different-node variants, so these are the only conflict shapes.
///
/// **Detection is SOUND, not complete**, and each narrowing costs only
/// diagnostics, never a wrong type:
///
/// - Sharing is decided by NODE IDENTITY of `draw` nodes reached through the
///   binding DAG ([`component_draw_nodes`]) — the intersection the maths doc §10
///   prescribes. Two components reaching the same `draw` node genuinely share it;
///   a component whose trace this walk cannot follow (a cross-module ref, a
///   lambda kernel, a depth-capped path) contributes no draws and so triggers
///   nothing.
/// - A component's own boundary SEVERS the walk ([`component_draw_nodes`]'s
///   `boundary` argument): §04 substitutes a boundary node with a fresh input in the
///   reified graph, so a draw at or beyond the boundary is not in that component's
///   trace and cannot be shared through it. Without the cut, `K2 = kernelof(a2, u =
///   u)` would look like a sharing non-binder of `u`'s own ancestor and be rejected,
///   though it shares nothing (`kernel-joint-w1-maths.md` §5, third bullet).
/// - Only a `Type::Measure` component counts as a proven non-binder ([`Binds`]). A
///   kernel component whose boundary this pass cannot read is unknown, not a
///   non-binder, so it is skipped rather than rejected.
/// - The conflict is reported only for a boundary target the shared node's own
///   subtree actually reaches ([`subtree_reaches_name`]) — "every ancestor of the
///   shared node that any component binds", not "any boundary the two components
///   happen to share". A boundary that is NOT an ancestor of the shared node is
///   legal (the retained node's parent is still single-valued), so widening the test
///   would reject a well-formed program.
///
/// The two directions are checked symmetrically, because "any component binds" makes
/// the obligation mutual: the binder may be either side, and only one of them
/// declares the ancestor in the W1 shape.
///
/// This is not the cross-component ancestry oracle `product_mass` deliberately
/// avoids: it only ever ADDS a diagnostic on a proven identity, and never
/// strengthens a mass class or a type on an unproven one.
fn diagnose_shared_node_input_names(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    components: &[(NodeId, &Type)],
) -> Option<Type> {
    /// One inspectable component: the `draw` nodes its trace reaches, and what it
    /// binds as boundary inputs.
    struct Traced {
        draws: Vec<NodeId>,
        binds: Binds,
    }
    let mut traced: Vec<Traced> = Vec::new();
    for (node, t) in components {
        let binds = match t {
            Type::Measure { .. } => Binds::Nothing,
            Type::Kernel { .. } => {
                let Some((reif, _)) = local_reification(inf, *node) else {
                    continue;
                };
                let Some(entries) = input_entries(inf, reif) else {
                    continue;
                };
                Binds::Declared(entries.iter().map(|(name, r)| (*name, r.name)).collect())
            }
            _ => continue,
        };
        let boundary: Vec<Symbol> = match &binds {
            Binds::Declared(entries) => entries.iter().map(|(_, target)| *target).collect(),
            Binds::Nothing => Vec::new(),
        };
        traced.push(Traced {
            draws: component_draw_nodes(inf, *node, &boundary),
            binds,
        });
    }
    for i in 0..traced.len() {
        for j in (i + 1)..traced.len() {
            for shared in traced[i]
                .draws
                .iter()
                .filter(|d| traced[j].draws.contains(d))
            {
                for (binder, other) in [(&traced[i], &traced[j]), (&traced[j], &traced[i])] {
                    let Binds::Declared(entries) = &binder.binds else {
                        continue;
                    };
                    for (input, target) in entries {
                        if !subtree_reaches_name(inf, *shared, *target) {
                            continue;
                        }
                        let counterpart = match &other.binds {
                            Binds::Declared(theirs) => {
                                theirs.iter().find(|(_, t)| t == target).map(|(n, _)| *n)
                            }
                            Binds::Nothing => None,
                        };
                        if counterpart == Some(*input) {
                            continue;
                        }
                        let target = inf.module.resolve(*target).to_string();
                        let mine = inf.module.resolve(*input).to_string();
                        let (message, summary) = match counterpart {
                            Some(theirs) => {
                                let theirs = inf.module.resolve(theirs).to_string();
                                (
                                    format!(
                                        "`joint` components share a stochastic node whose \
                                         boundary ancestor `{target}` is bound under different \
                                         input names (`{mine}` and `{theirs}`): the shared \
                                         node's parent has no well-defined value where the two \
                                         inputs differ, so every sharing component must bind it \
                                         under the same name (spec §06 `joint`)"
                                    ),
                                    "joint kernel components disagree on a shared node's input \
                                     name",
                                )
                            }
                            // The non-binder's KIND is read off `other.binds`, not assumed:
                            // a measure component binds nothing by being nullary, while a
                            // kernel component that simply omits the ancestor from its own
                            // boundary is the same clause and a different mistake. Naming
                            // the wrong one sends the reader looking for a measure component
                            // that is not there.
                            None => {
                                // The non-binder's KIND names the component, and the reason
                                // follows the verb rather than interrupting it — reading
                                // "while a measure component, which binds nothing, binds it
                                // under no name" doubles the verb and reads as a typo.
                                let (who, why) = match &other.binds {
                                    Binds::Nothing => (
                                        "a measure component",
                                        "measure components are nullary and declare no boundary \
                                         inputs",
                                    ),
                                    Binds::Declared(_) => (
                                        "another kernel component",
                                        "its own boundary omits that ancestor",
                                    ),
                                };
                                (
                                    format!(
                                        "`joint` components share a stochastic node whose \
                                         boundary ancestor `{target}` one component binds as \
                                         `{mine}` while {who} binds it under no name ({why}): \
                                         the shared node would carry the applied input's law \
                                         and the ambient `{target}`'s law at once, so every \
                                         sharing component must bind that ancestor under the \
                                         same name (spec §06 `joint`)"
                                    ),
                                    "joint components disagree on a shared node's ancestry: one \
                                     binds it under no name",
                                )
                            }
                        };
                        inf.diags.push(crate::Diagnostic::error_at(id, message));
                        return Some(Type::Failed(summary.into()));
                    }
                }
            }
        }
    }
    None
}

/// The `draw` nodes a `joint` component's subtree reaches, by NODE IDENTITY,
/// following self-module refs the way [`joint_component_is_trace_clean`] does and
/// under the same `depth` cap. A stochastic node enters a `joint`'s composed trace
/// only through a reified component or a stochastic constructor parameter (§04
/// "Trace of the reified law", §06 `joint` entry), and both channels are subtrees
/// of the component, so both are covered by one walk.
///
/// Node identity is what makes the answer meaningful: `u ~ Normal(…)` has ONE
/// `draw` node in the binding DAG, so two components that both reach it through
/// `(%ref self u)` reach the same `NodeId`, while two independent draws never do.
/// The result is deliberately a proof of sharing only — a path this walk cannot
/// follow yields fewer draws, never a spurious one.
///
/// `boundary` names the component's OWN boundary targets, and the walk stops at each
/// of them without recording anything beyond. §04 "Specifying reification
/// boundaries" is why: "A specified boundary node `a` can be thought of as being
/// substituted with a new node, generated via `elementof(valueset(a))`, in the
/// reified graph", so a draw at or past the boundary is not in this component's trace
/// and cannot be shared through it.
fn component_draw_nodes(
    inf: &Inferencer<'_, '_>,
    node: NodeId,
    boundary: &[Symbol],
) -> Vec<NodeId> {
    fn walk(
        inf: &Inferencer<'_, '_>,
        node: NodeId,
        boundary: &[Symbol],
        depth: u32,
        seen: &mut std::collections::HashSet<NodeId>,
        out: &mut Vec<NodeId>,
    ) {
        if depth > 64 || !seen.insert(node) {
            return;
        }
        if let Node::Ref(r) = inf.module.node(node) {
            if r.ns == RefNs::SelfMod && !boundary.contains(&r.name) {
                if let Some(b) = inf.module.binding_by_name(r.name) {
                    let rhs = inf.module.binding(b).rhs;
                    walk(inf, rhs, boundary, depth + 1, seen, out);
                }
            }
            return;
        }
        if let Node::Call(c) = inf.module.node(node) {
            if let CallHead::Builtin(op) = c.head {
                if inf.module.resolve(op) == "draw" {
                    out.push(node);
                }
            }
        }
        let mut children = Vec::new();
        inf.module
            .for_each_child(node, |child| children.push(child));
        for child in children {
            walk(inf, child, boundary, depth + 1, seen, out);
        }
    }
    let mut out = Vec::new();
    walk(
        inf,
        node,
        boundary,
        0,
        &mut std::collections::HashSet::new(),
        &mut out,
    );
    out
}

/// Does the subtree at `root` reach `(%ref self name)`, following self-module refs
/// under the same `depth` cap as [`component_draw_nodes`]? Used to restrict the Q1
/// conflict to a boundary target that is genuinely an ANCESTOR of the shared node.
fn subtree_reaches_name(inf: &Inferencer<'_, '_>, root: NodeId, name: Symbol) -> bool {
    fn walk(
        inf: &Inferencer<'_, '_>,
        node: NodeId,
        name: Symbol,
        depth: u32,
        seen: &mut std::collections::HashSet<NodeId>,
    ) -> bool {
        if depth > 64 || !seen.insert(node) {
            return false;
        }
        if let Node::Ref(r) = inf.module.node(node) {
            if r.ns != RefNs::SelfMod {
                return false;
            }
            if r.name == name {
                return true;
            }
            let Some(b) = inf.module.binding_by_name(r.name) else {
                return false;
            };
            let rhs = inf.module.binding(b).rhs;
            return walk(inf, rhs, name, depth + 1, seen);
        }
        let mut children = Vec::new();
        inf.module
            .for_each_child(node, |child| children.push(child));
        children
            .into_iter()
            .any(|child| walk(inf, child, name, depth + 1, seen))
    }
    walk(inf, root, name, 0, &mut std::collections::HashSet::new())
}

/// The output measure of an APPLIED kernel `joint` — `joint(K1, K2, …)(a)`.
/// Spec §06 makes this the `joint` of the component output measures at that input
/// ("At each input point the result is the `joint` of the component output
/// measures"), so the variate is the same record (keyword) or `cat` (positional)
/// merge as the measure case, and this is where Q2's record variate becomes
/// visible — `Type::Kernel` has no slot for it.
///
/// The mass is left `%deferred` for the caller to fill from the kernel's own
/// class, which §11 defines as uniform over all inputs. `None` when `callee` is
/// not a local `joint` call or a component's variate is not statically
/// resolvable, so the application stays `%deferred` rather than guessing a shape.
fn kernel_joint_result_type(inf: &mut Inferencer<'_, '_>, callee: NodeId) -> Option<Type> {
    let mut node = callee;
    let call = loop {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(c) => {
                let CallHead::Builtin(op) = c.head else {
                    return None;
                };
                if inf.module.resolve(op) != "joint" {
                    return None;
                }
                break c.clone();
            }
            _ => return None,
        }
    };
    // A mixed spelling is a static error ([`kernel_joint_type`]), so it never
    // reaches here with a `Type::Kernel` callee. Declined rather than read, so the
    // component-dropping split exists in one place instead of two.
    if !call.args.is_empty() && !call.named.is_empty() {
        return None;
    }
    let components: Vec<(Option<Symbol>, NodeId)> = if call.named.is_empty() {
        call.args.iter().map(|&a| (None, a)).collect()
    } else {
        call.named
            .iter()
            .map(|na| (Some(na.name), na.value))
            .collect()
    };
    if components.is_empty() {
        return None;
    }
    let mut variates = Vec::with_capacity(components.len());
    for (name, component) in components {
        let ty = inf.lookup_type(component).cloned()?;
        let variate = component_variate(inf, component, &ty)?;
        variates.push((name, variate));
    }
    let domain = if variates.iter().all(|(name, _)| name.is_some()) {
        Type::Record(
            variates
                .into_iter()
                .map(|(name, v)| (name.expect("all named"), v))
                .collect(),
        )
    } else {
        cat_compose(&variates.into_iter().map(|(_, v)| v).collect::<Vec<_>>())
    };
    Some(Type::Measure {
        domain: Box::new(domain),
        mass: Mass::Deferred,
    })
}

/// `ksuperpose(kernel, weights)` (spec §06 "Additive superposition") — the
/// weighted-superposition LIFT. The call itself is a kernel, so this rule types
/// the CURRIED form; the applied form is [`ksuperpose_result_type`].
///
/// §06: "`ksuperpose(kernel, weights)` is itself a kernel; applied to a
/// parameter family it yields the mixture $\nu = \sum_i w_i\,\kappa(\theta_i)$".
///
/// **Inputs.** §04's arity row gives `ksuperpose` "Two distinguished inputs (the
/// kernel and the weight vector); the resulting kernel is applied separately to
/// the parameter family". The family is passed "as to `broadcast`", so the
/// lifted kernel's parameter NAMES are the component kernel's own — a reified
/// component declares them in its type, a bare constructor in §08's parameter
/// column ([`ksuperpose_component_inputs`]). An unreadable component leaves the
/// list empty, which makes `local_kernel_inputs` decline and skips
/// [`user_arity_check`] rather than inventing an arity of zero.
///
/// **Mass** is `weighted`'s rule over a vector of weights, not `superpose`'s
/// over a component list: §06 puts the total mass at
/// $\sum_i w_i\,\mathrm{totalmass}(\kappa(\theta_i))$, "which is $\sum_i w_i$
/// for a Markov `kernel`", and "need not be normalized, so the result is
/// generally unnormalized". So a Markov component demotes to `%finite`, exactly
/// as `weighted(w, M)` demotes a normalized base under a fixed scalar weight.
/// Weights this pass cannot read as a fixed collection give `%unknown`, the same
/// distrust `weighted` applies to a non-fixed weight — an unread weight could be
/// infinite, and §06's all-zero case makes it the zero measure, which is not
/// `%finite`'s guarantee either.
fn ksuperpose_type(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    // Two DISTINGUISHED inputs (§04), so neither is a keyword-spelled parameter
    // and neither is variadic — unlike `superpose`, whose components are.
    if !named.is_empty() || args.len() != 2 {
        let got = args.len() + named.len();
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            format!(
                "`ksuperpose` takes 2 positional arguments — the kernel and the \
                 weight vector (spec §04 \"Calling conventions\": two distinguished \
                 inputs) — got {got}"
            ),
        ));
        return Type::Failed("ksuperpose takes 2 positional arguments".into());
    }
    let (weights_node, weights_ty, weights_phase) = args[1].clone();
    if let Some(failed) = ksuperpose_weights_check(inf, weights_node, &weights_ty) {
        return failed;
    }
    let (component_node, component_ty, _) = args[0].clone();
    let inputs = ksuperpose_component_inputs(inf, component_node, &component_ty);
    let mass = ksuperpose_mass(
        ksuperpose_component_mass(inf, component_node, &component_ty),
        &weights_ty,
        weights_phase,
    );
    Type::Kernel { inputs, mass }
}

/// Reject a `ksuperpose` weight argument that cannot be a weight vector.
/// §06 fixes `N` as "the length of `weights`", and the family rule measures every
/// collection argument against that one axis, so a scalar or a multi-axis weight
/// argument leaves `N` undefined rather than merely unknown. A `%deferred` or
/// `%any` weight type is admitted: §06 says `N` "need not be statically known",
/// and non-negativity is a runtime domain condition no type slot records.
fn ksuperpose_weights_check(inf: &mut Inferencer<'_, '_>, node: NodeId, ty: &Type) -> Option<Type> {
    let complaint = match ty {
        // `flatten_dims` counts a nested array's axes as well as a multi-dim
        // `shape`, so `[[…], […]]` is rejected as readily as a matrix.
        Type::Array { .. } => match axis_count(ty) {
            1 => return None,
            axes => format!("an array with {axes} axes"),
        },
        Type::TVector { .. } | Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_) => {
            return None;
        }
        Type::Table { .. } => "a table".to_string(),
        other => match non_measure_kind(other) {
            Some(kind) => kind.to_string(),
            None => "a measure".to_string(),
        },
    };
    inf.diags.push(crate::Diagnostic::error_at(
        node,
        format!(
            "`ksuperpose`'s weights must be a vector (spec §06: \"The number of \
             components $N$ is the length of `weights`\"); got {complaint} instead"
        ),
    ));
    Some(Type::Failed("ksuperpose weights are not a vector".into()))
}

/// The parameter names the lifted kernel declares — the COMPONENT kernel's own,
/// since §06 passes the family "as to `broadcast`" and `broadcast` binds a
/// positional data-arg to the head's ordered parameter name. Empty when this pass
/// cannot read them, which [`local_kernel_inputs`] treats as "no declared list"
/// rather than as a nullary callable.
fn ksuperpose_component_inputs(
    inf: &mut Inferencer<'_, '_>,
    node: NodeId,
    ty: &Type,
) -> Box<[Symbol]> {
    if let Type::Kernel { inputs, .. } = ty {
        return inputs.clone();
    }
    let Some(name) = ksuperpose_constructor_name(inf, node) else {
        return Box::new([]);
    };
    let Some(params) = crate::constructor_param_names(&name) else {
        return Box::new([]);
    };
    params.iter().map(|p| inf.module.intern(p)).collect()
}

/// The component's own measure-constructor name, following `%ref self` hops to a
/// bare `Const` head. `ksuperpose(Normal, w)` and `K = Normal` /
/// `ksuperpose(K, w)` both reach `"Normal"`; a §09 member ref
/// (`hepphys.ContinuedPoisson`) reaches its member name, which
/// `constructor_param_names` also resolves.
fn ksuperpose_constructor_name(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<String> {
    let mut node = node;
    for _ in 0..64 {
        match inf.module.node(node) {
            Node::Const(sym) => return Some(inf.module.resolve(*sym).to_string()),
            Node::Ref(Ref {
                ns: RefNs::Module(_),
                name,
            }) => return Some(inf.module.resolve(*name).to_string()),
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            _ => return None,
        }
    }
    None
}

/// The component's total-mass class. [`component_mass`] covers a measure or a
/// trusted-head kernel; a BARE constructor (`ksuperpose(Normal, w)`) is neither,
/// and reading it `%unknown` would lose the one class §06's mass sentence names
/// explicitly ("$\sum_i w_i$ for a Markov `kernel`"). A recognized §08/§09
/// distribution is a probability measure, which is `fill_mass`'s own catchall
/// reading of a constructor head; `Dirac` is a point mass, also normalized.
fn ksuperpose_component_mass(inf: &Inferencer<'_, '_>, node: NodeId, ty: &Type) -> Mass {
    match component_mass(inf, node, ty) {
        Mass::Unknown => match ksuperpose_constructor_name(inf, node) {
            Some(name) if crate::distribution_param_names(&name).is_some() => Mass::Normalized,
            Some(name) if name == "Dirac" => Mass::Normalized,
            _ => Mass::Unknown,
        },
        known => known,
    }
}

/// §06's mass rule for the lift, shaped after `fill_mass`'s `weighted` arm.
fn ksuperpose_mass(component: Mass, weights_ty: &Type, weights_phase: Phase) -> Mass {
    if component == Mass::Null {
        // Every component is the zero measure, so every weighted term is too.
        return Mass::Null;
    }
    let weights_readable = matches!(
        (weights_ty, weights_phase),
        (Type::Array { .. } | Type::TVector { .. }, Phase::Fixed)
    );
    if !weights_readable {
        return Mass::Unknown;
    }
    match component {
        // Generally unnormalized: the weights need not sum to one.
        Mass::Normalized | Mass::Finite => Mass::Finite,
        Mass::LocallyFinite => Mass::LocallyFinite,
        _ => Mass::Unknown,
    }
}

/// The output measure of an APPLIED `ksuperpose` — `ksuperpose(K, w)(θ)`.
///
/// §06 makes it the mixture $\sum_i w_i\,\kappa(\theta_i)$ over the components'
/// SHARED variate, so — unlike `broadcast`, whose applied form is the independent
/// product over the family and whose domain therefore gains an axis — the family
/// axis is CONTRACTED here and the domain is the component's per-cell variate.
///
/// This is also where §06's one-axis family rule is enforced, because the family
/// arguments are the arguments of the APPLICATION, not of the lift: "the family
/// has a single axis: along it every collection argument must have size $N$ or be
/// singular (size one) … A table counts as having one axis, its rows; a
/// collection argument with more than one axis is a static error."
///
/// `None` (→ the caller leaves the application `%deferred`) when the component's
/// per-cell variate is not statically resolvable, never a guessed shape.
fn ksuperpose_result_type(
    inf: &mut Inferencer<'_, '_>,
    callee: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    let call = ksuperpose_callee(inf, callee)?;
    let weights = *call.args.get(1)?;
    if let Some(failed) = ksuperpose_family_check(inf, callee, weights, args, named) {
        return Some(failed);
    }
    let component = *call.args.first()?;
    let cell = ksuperpose_cell_variate(inf, component, args, named)?;
    Some(Type::Measure {
        domain: Box::new(cell),
        // `user_call_type` fills this from the lifted kernel's own class, which
        // §11 defines as "uniform over all inputs".
        mass: Mass::Deferred,
    })
}

/// The `ksuperpose` call a callee resolves to, following `%ref self` hops.
/// `None` for any other callee, so the applied-kernel chain in
/// [`user_call_type`] falls through unchanged.
fn ksuperpose_callee(inf: &Inferencer<'_, '_>, callee: NodeId) -> Option<Call> {
    let mut node = callee;
    for _ in 0..64 {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(c) => {
                let CallHead::Builtin(op) = c.head else {
                    return None;
                };
                if inf.module.resolve(op) != "ksuperpose" {
                    return None;
                }
                if c.args.len() != 2 {
                    return None;
                }
                return Some(c.clone());
            }
            _ => return None,
        }
    }
    None
}

/// §06's one-axis family rule. Each family argument is measured against `N`, the
/// weights' own length: a collection must be one-axis and size `N` or singular,
/// a non-collection is held constant. A `Dim::Dynamic` extent on either side is
/// admitted — §06 says `N` "need not be statically known", so a dynamic length
/// is a runtime check, not a static error.
fn ksuperpose_family_check(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    weights: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    let n = match inf.lookup_type(weights) {
        Some(t @ Type::Array { .. }) if axis_count(t) == 1 => flatten_dims(t)[0],
        Some(Type::TVector { len, .. }) => *len,
        _ => Dim::Dynamic,
    };
    let family: Vec<(NodeId, Type)> = args
        .iter()
        .map(|(node, t, _)| (*node, t.clone()))
        .chain(named.iter().map(|(_, node, t, _)| (*node, t.clone())))
        .collect();
    let mut failed = None;
    for (node, ty) in family {
        // A table's rows ARE the one axis (§06), so its `nrows` is the extent
        // and it can never be the multi-axis error.
        let extent = match &ty {
            Type::Array { .. } => match axis_count(&ty) {
                1 => flatten_dims(&ty)[0],
                axes => {
                    inf.diags.push(crate::Diagnostic::error_at(
                        node,
                        format!(
                            "a `ksuperpose` family argument with more than one axis is a \
                             static error (spec §06: \"a collection argument with more \
                             than one axis is a static error\"); got an array with \
                             {axes} axes"
                        ),
                    ));
                    failed = Some(Type::Failed(
                        "ksuperpose family argument is multi-axis".into(),
                    ));
                    continue;
                }
            },
            Type::TVector { len, .. } => *len,
            Type::Table { nrows, .. } => *nrows,
            // Held constant across the components (§06).
            _ => continue,
        };
        let (Dim::Static(got), Dim::Static(want)) = (extent, n) else {
            continue;
        };
        if got != want && got != 1 {
            inf.diags.push(crate::Diagnostic::error_at(
                node,
                format!(
                    "a `ksuperpose` family argument must have size {want} — the length \
                     of `weights` — or be singular (spec §06); got size {got}"
                ),
            ));
            failed = Some(Type::Failed(
                "ksuperpose family argument size does not match the weights".into(),
            ));
        }
    }
    let _ = id;
    failed
}

/// How many axes a collection type has, counting a nested array's axes as well
/// as a multi-dimensional `shape` — §04 Broadcasting's "number of axes", which
/// [`flatten_dims`] already computes for `aggregate`.
fn axis_count(t: &Type) -> usize {
    flatten_dims(t).len()
}

/// The component kernel's PER-CELL variate: the mixture's own domain, since the
/// components share one variate. Each family argument contributes its element
/// type, exactly as `broadcast_type`'s `cell_arg` does; a non-collection rides
/// along whole.
fn ksuperpose_cell_variate(
    inf: &mut Inferencer<'_, '_>,
    component: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    // One axis stripped per family argument, as `broadcast_type`'s `cell_arg`
    // does. A table's row IS its cell (§06: "A table counts as having one axis,
    // its rows"), so it strips to the record of its columns.
    let cell_arg = |t: &Type| match t {
        Type::Array { elem, .. } => elem.as_ref().clone(),
        Type::TVector { elem, .. } => elem.as_ref().clone(),
        Type::Table { columns, .. } => Type::Record(columns.clone()),
        other => other.clone(),
    };
    let cell_args: Vec<ArgInfo> = args.iter().map(|(n, t, p)| (*n, cell_arg(t), *p)).collect();
    let cell_named: Vec<NamedInfo> = named
        .iter()
        .map(|(s, n, t, p)| (*s, *n, cell_arg(t), *p))
        .collect();
    // A reified component reaches its variate by substitution, exactly as
    // `broadcast_type`'s `Type::Kernel` head does.
    if let Some(ty) = substituted_result(inf, component, &cell_args, &cell_named)
        .map(|(ty, _)| ty)
        .or_else(|| reified_result_type(inf, component))
    {
        return match ty {
            Type::Measure { domain, .. } if !matches!(*domain, Type::Deferred) => Some(*domain),
            Type::Measure { .. } | Type::Deferred => None,
            value_ty => Some(value_ty),
        };
    }
    // A §09 member reference carries its catalogue sig; a bare builtin
    // constructor reaches its domain by name. Both mirror `broadcast_type`.
    if let Some(sig) = inf.module_catalogue_ref(component).map(|c| c.sig.clone()) {
        return match catalogue_lower(&mut *inf.module, &sig, &cell_args).0 {
            Type::Measure { domain, .. } => Some(*domain),
            _ => None,
        };
    }
    let name = ksuperpose_constructor_name(inf, component)?;
    if let Some(domain) = distribution_domain(inf, &name, &cell_args, &cell_named) {
        return Some(domain);
    }
    // `Dirac` is a §06 FUNDAMENTAL measure, deliberately outside the §08
    // distribution catalogue, so `distribution_domain` declines it. Its variate
    // is its `value` argument — here the per-cell element of the family's
    // `value` column, which is what makes §08's
    // `normalize(ksuperpose(Dirac, p)(value = labels))` a measure over the
    // labels' own type. `Lebesgue`/`Counting` take a support SET rather than a
    // point, and no family reading of that is settled, so they stay `%deferred`.
    if name == "Dirac" {
        return cell_named
            .iter()
            .find(|(n, _, _, _)| inf.module.resolve(*n) == "value")
            .map(|(_, _, t, _)| t.clone())
            .or_else(|| cell_args.first().map(|(_, t, _)| t.clone()));
    }
    None
}

/// `functionof` / `kernelof` (spec §04 reification, §11 reified callables).
/// A `functionof` whose body is a measure *is* a kernel.
fn reification_type(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    call: &Call,
    name: &str,
    args: &[ArgInfo],
) -> Type {
    // Boundary entries whose target ref is a `%local` placeholder — that entry
    // IS the placeholder's declaration (spec §04), under either origin tag. The
    // auto-trace itself only ever records `elementof` leaves, but FlatPIR may
    // carry an explicit entry list under `%autoinputs`, and such an entry
    // declares its placeholder exactly as a `%specinputs` entry does.
    let mut declared: Vec<Symbol> = Vec::new();
    let inputs: Box<[Symbol]> = match call.inputs.as_ref() {
        Some(Inputs::Spec(entries)) => {
            // Reification is module-local (spec §04): a boundary may not designate
            // a loaded-module binding — reification never crosses a module
            // boundary, even explicitly. Use the dependency's callables/values.
            if let Some((_, r)) = entries
                .iter()
                .find(|(_, r)| matches!(r.ns, RefNs::Module(_)))
            {
                inf.diags.push(crate::Diagnostic::error_at(
                    id,
                    format!(
                        "reification is module-local: boundary `{}` designates a \
                         loaded-module binding — reification cannot cross a module \
                         boundary (spec §04); use the dependency's callables/values instead",
                        inf.module.resolve(r.name)
                    ),
                ));
                return Type::Failed("cross-module reification boundary".into());
            }
            declared.extend(
                entries
                    .iter()
                    .filter(|(_, r)| r.ns == RefNs::Local)
                    .map(|(_, r)| r.name),
            );
            entries.iter().map(|(n, _)| *n).collect()
        }
        Some(Inputs::Auto) => match inf.module.auto_inputs_of(id) {
            Some(entries) => {
                declared.extend(
                    entries
                        .iter()
                        .filter(|(_, r)| r.ns == RefNs::Local)
                        .map(|(_, r)| r.name),
                );
                entries.iter().map(|(n, _)| *n).collect()
            }
            None => {
                // §04 auto-trace: discover the body's `elementof` parametric
                // leaves (canonical-sorted by name) and fill the side-table, so
                // the reification types as a kernel/function over those inputs.
                let Some((body, _, _)) = args.first() else {
                    return Type::Deferred;
                };
                let (entries, cross_module) = inf.collect_auto_inputs(*body);
                if cross_module {
                    // Reification is module-local (spec §04): a parameterized value
                    // reached through a loaded-module reference cannot become an
                    // input. (A cross-module callable/value may be USED — applied or
                    // referenced — just not taken as a reified input.)
                    inf.diags.push(crate::Diagnostic::error_at(
                        id,
                        "reification is module-local: this depends on a parameterized \
                         value from a loaded module, which cannot become an input \
                         (spec §04); use the dependency's callables/values instead, or \
                         reify within the module that defines it",
                    ));
                    return Type::Failed("cross-module reification".into());
                }

                let names: Box<[Symbol]> = entries.iter().map(|(n, _)| *n).collect();
                inf.module.set_auto_inputs(id, entries.into());
                names
            }
        },
        None => unreachable!("reification_type called only when inputs are present"),
    };
    // §04 *Specifying reification boundaries*: "Boundary input names must be
    // distinct — a repeated name is a static error, which likewise forbids a
    // lambda or named function from repeating an argument name." The §05 sugars
    // (`f(a, a) = …`, `(a, a) -> …`) lower to this same boundary list, so one
    // check here covers every reified form.
    let mut seen: Vec<Symbol> = Vec::with_capacity(inputs.len());
    let mut repeated: Vec<Symbol> = Vec::new();
    for n in inputs.iter() {
        if seen.contains(n) {
            if !repeated.contains(n) {
                repeated.push(*n);
            }
        } else {
            seen.push(*n);
        }
    }
    if !repeated.is_empty() {
        for n in &repeated {
            inf.diags.push(crate::Diagnostic::error_at(
                id,
                format!(
                    "boundary input `{}` is declared more than once (spec §04 Specifying \
                     reification boundaries: \"Boundary input names must be distinct — a \
                     repeated name is a static error, which likewise forbids a lambda or \
                     named function from repeating an argument name\"); give each input a \
                     distinct name",
                    inf.module.resolve(*n)
                ),
            ));
        }
        return Type::Failed("repeated boundary input name".into());
    }
    // §04 *Placeholders and holes*, the front door: a placeholder this
    // reification's body reaches and its boundary does not declare is a static
    // error. Unenforced it reaches the determiniser as a dangling `(%ref %local
    // …)` inside a scored density (`density::lower_reified_measure` screens the
    // same hole from behind).
    if let Some((body, _, _)) = args.first() {
        let undeclared = undeclared_placeholders(inf.module, *body, &declared);
        if !undeclared.is_empty() {
            for (ph, at) in undeclared {
                let name = inf.module.resolve(ph).to_string();
                let kw = name.trim_matches('_').to_string();
                inf.diags.push(crate::Diagnostic::error_at(
                    at,
                    format!(
                        "placeholder `{name}` appears in the reified expression but no boundary \
                         input declares it (spec §04 Placeholders and holes: \"All placeholders \
                         must appear both in the expression to be reified and the boundary input \
                         keyword arguments\"); declare it as `{kw} = {name}`"
                    ),
                ));
            }
            return Type::Failed("undeclared placeholder".into());
        }
    }
    let body_ty = args.first().map(|(_, t, _)| t);
    match (name, body_ty) {
        // `kernelof` reifies the LAW of a value-typed body — a probability
        // measure per input, i.e. a Markov kernel. Its body must BE a value:
        // §04 "Kernels and `kernelof`" says it "reifies (typically stochastic)
        // value nodes to Markov kernels. `x` must not be a measure", and
        // flatppl-design#73 adds the reason — "since `functionof` already reifies
        // a measure node to a kernel directly". A measure-layer body is therefore
        // a static error, not something to wrap.
        ("kernelof", Some(body @ (Type::Measure { .. } | Type::Kernel { .. }))) => {
            let anchor = args.first().map_or(id, |(n, _, _)| *n);
            // §04's sentence names the measure case; a KERNEL body fails the same
            // clause one step earlier (it is not a value node either) and #73's
            // `functionof` reason covers it identically. Named separately so the
            // diagnostic does not tell a user their kernel is a measure.
            let (what, fix) = match body {
                Type::Measure { .. } => (
                    "a measure",
                    "use `functionof` to reify a measure node to a kernel directly, \
                     or pass the value you meant to take the law of",
                ),
                _ => (
                    "a kernel",
                    "a kernel is already a reified law — pass the value node you meant \
                     to reify, or drop the outer `kernelof`",
                ),
            };
            inf.diags.push(crate::Diagnostic::error_at(
                anchor,
                format!(
                    "`kernelof` reifies value nodes, but this argument is {what} \
                     (spec §04: `x` must not be a measure); {fix}"
                ),
            ));
            Type::Failed("kernelof of a measure-layer argument".into())
        }
        ("kernelof", _) => Type::Kernel {
            inputs,
            mass: Mass::Normalized,
        },
        ("functionof", Some(Type::Measure { mass, .. })) => Type::Kernel {
            inputs,
            mass: *mass,
        },
        ("functionof", _) => Type::Function { inputs },
        _ => Type::Deferred,
    }
}

/// Every `%local` placeholder the reified expression at `body` reaches that
/// `declared` does not list, as `(placeholder, first occurrence)` pairs in walk
/// order. Spec §04 *Placeholders and holes*: "All placeholders must appear both
/// in the expression to be reified and the boundary input keyword arguments."
/// The operative rule: a boundary entry targeting the placeholder declares it,
/// under either origin tag — a `%specinputs` entry or an `%autoinputs` entry.
///
/// A nested `functionof`/`kernelof` is its OWN placeholder scope (§04: "The
/// scope of a placeholder is the nearest enclosing `functionof` or `kernelof`"),
/// so the walk stops at one: a placeholder free there is that reification's
/// error, reported when inference reaches it (children are traced first).
/// Self-refs ARE followed, so a placeholder one binding away is still caught.
fn undeclared_placeholders(
    module: &flatppl_core::Module,
    body: NodeId,
    declared: &[Symbol],
) -> Vec<(Symbol, NodeId)> {
    fn walk(
        module: &flatppl_core::Module,
        id: NodeId,
        declared: &[Symbol],
        found: &mut Vec<(Symbol, NodeId)>,
        visited: &mut std::collections::HashSet<NodeId>,
    ) {
        if !visited.insert(id) {
            return;
        }
        match module.node(id) {
            Node::Ref(r) if r.ns == RefNs::Local => {
                if !declared.contains(&r.name) && !found.iter().any(|(p, _)| *p == r.name) {
                    found.push((r.name, id));
                }
                return;
            }
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                if let Some(b) = module.binding_by_name(r.name) {
                    let rhs = module.binding(b).rhs;
                    walk(module, rhs, declared, found, visited);
                }
                return;
            }
            Node::Call(c)
                if c.inputs.is_some()
                    && matches!(c.head, CallHead::Builtin(h)
                        if matches!(module.resolve(h), "functionof" | "kernelof")) =>
            {
                return;
            }
            _ => {}
        }
        for child in module.node(id).children() {
            walk(module, child, declared, found, visited);
        }
    }
    let mut found = Vec::new();
    walk(
        module,
        body,
        declared,
        &mut found,
        &mut std::collections::HashSet::new(),
    );
    found
}

/// `likelihoodof(K, obs)` — inputs ride over from the kernel; the obstype is
/// the kernel's measure domain, recovered by looking through to the reified
/// body (spec §11 `%likelihood`).
///
/// When the kernel comes from `disintegrate` (via `fk, prior = disintegrate(sel,
/// joint)`, which desugars to `fk = get(__synth, 1)`), `reified_result_type`
/// returns `None` because `get` is not a reification. In that case we fall back
/// to `disintegrate_kernel_obstype`, which follows the `get` → ref →
/// `disintegrate` chain and re-derives the SELECTED-variate record type (the
/// mirror of `disintegrate_type`'s complement computation, keeping selected
/// fields). Spec §06 "Structural disintegration".
fn likelihood_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Type {
    let Some((k_node, Type::Kernel { inputs, .. }, _)) = args.first() else {
        return Type::Deferred;
    };
    let inputs = inputs.clone();
    // A `functionof`-of-measure body exposes its domain; a `kernelof` body is
    // the random *value* itself, so its type is the observation domain.
    match reified_result_type(inf, *k_node) {
        Some(Type::Measure { domain, .. }) => Type::Likelihood {
            inputs,
            obstype: domain,
        },
        Some(Type::Deferred) | None => {
            // Fall back: check if this kernel came from `disintegrate` and
            // recover the selected-variate obstype from the joint's record domain.
            match disintegrate_kernel_obstype(inf, *k_node) {
                Some(obstype) => Type::Likelihood {
                    inputs,
                    obstype: Box::new(obstype),
                },
                None => Type::Deferred,
            }
        }
        Some(value_ty) => Type::Likelihood {
            inputs,
            obstype: Box::new(value_ty),
        },
    }
}

/// Recover the obstype for a kernel that came from `disintegrate`. The
/// desugaring of `fk, prior = disintegrate(sel, joint)` produces
/// `fk = get(__synth, 1)` where `__synth` is bound to the `disintegrate`
/// call. We follow the `get` → ref → `disintegrate` chain and re-derive the
/// SELECTED-variate record (the mirror of `disintegrate_type`'s complement
/// computation, but keeping the selected fields instead of the complement).
/// Returns `None` when the chain is absent or the joint is not a record-domain
/// measure with a static selector (honest — never fabricates a type).
fn disintegrate_kernel_obstype(inf: &mut Inferencer<'_, '_>, mut node: NodeId) -> Option<Type> {
    // Follow any self-module refs on the kernel node.
    loop {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let b = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(b).rhs;
            }
            _ => break,
        }
    }
    // Expect: `get(<tuple_ref>, 1)` — the first component of the disintegrate
    // tuple (1-based index). The second component is the marginal.
    let Node::Call(get_call) = inf.module.node(node).clone() else {
        return None;
    };
    if !matches!(get_call.head, CallHead::Builtin(op) if inf.module.resolve(op) == "get") {
        return None;
    }
    // arg[0] = tuple ref, arg[1] = index literal 1 (1-based first component)
    let (tuple_arg, idx_arg) = (
        get_call.args.first().copied()?,
        get_call.args.get(1).copied()?,
    );
    if !matches!(inf.module.node(idx_arg), Node::Lit(Scalar::Int(1))) {
        return None;
    }
    // Follow the tuple ref to the disintegrate call.
    let mut tuple_node = tuple_arg;
    loop {
        match inf.module.node(tuple_node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let b = inf.module.binding_by_name(r.name)?;
                tuple_node = inf.module.binding(b).rhs;
            }
            _ => break,
        }
    }
    let Node::Call(dis_call) = inf.module.node(tuple_node).clone() else {
        return None;
    };
    if !matches!(dis_call.head, CallHead::Builtin(op) if inf.module.resolve(op) == "disintegrate") {
        return None;
    }
    // Recover the selector field names (arg[0]) and the joint's record domain (arg[1]).
    let sel_node = *dis_call.args.first()?;
    let joint_node = *dis_call.args.get(1)?;
    let sel = selector_field_names(inf, sel_node)?;
    let joint_ty = inf.lookup_type(joint_node)?.clone();
    let domain = match &joint_ty {
        Type::Measure { domain, .. } => domain.as_ref(),
        _ => return None,
    };
    let Type::Record(fields) = domain else {
        return None;
    };
    // Keep the SELECTED fields (the forward kernel's output variate — spec §06).
    let is_sel = |n: &Symbol| sel.iter().any(|s| inf.module.resolve(*n) == &**s);
    let all_present = sel
        .iter()
        .all(|s| fields.iter().any(|(n, _)| inf.module.resolve(*n) == &**s));
    if !all_present {
        return None;
    }
    let selected: Vec<(Symbol, Type)> = fields.iter().filter(|(n, _)| is_sel(n)).cloned().collect();
    if selected.is_empty() {
        return None;
    }
    Some(Type::Record(selected.into()))
}

/// `joint_likelihood(L1, L2, …)` — combine likelihoods by multiplying densities
/// (spec §06). It is *defined* to equal `likelihoodof(joint(model1, …), cat(obs1,
/// …))`, so the combined inputs are the union of the component inputs (order-
/// preserving) and the combined obstype is the §06 cat-composition of the
/// component obstypes — NOT a tuple. Any non-likelihood (or `%deferred`) argument
/// defers the whole result.
fn joint_likelihood_type(args: &[ArgInfo]) -> Type {
    if args.is_empty() {
        return Type::Deferred;
    }
    let mut inputs: Vec<Symbol> = Vec::new();
    let mut obstypes: Vec<Type> = Vec::with_capacity(args.len());
    for (_, t, _) in args {
        let Type::Likelihood {
            inputs: li,
            obstype,
        } = t
        else {
            return Type::Deferred;
        };
        for name in li.iter() {
            if !inputs.contains(name) {
                inputs.push(*name);
            }
        }
        obstypes.push((**obstype).clone());
    }
    Type::Likelihood {
        inputs: inputs.into(),
        obstype: Box::new(cat_compose(&obstypes)),
    }
}

/// The spec §06 "same shape class" composition for `cat`-joined variates: the
/// type of `cat(x1, x2, …)` when the components share a shape class. Used for the
/// obstype of [`joint_likelihood_type`] (the joint observation is `cat(obs…)`);
/// the same rule is what a future `cat` / positional-`joint` / `jointchain`
/// domain rule needs.
///
/// - all scalars    → a length-`n` vector (component scalars promoted);
/// - all 1-D arrays → one concatenated 1-D array (lengths summed, `%dynamic` if
///   any is dynamic; elements promoted);
/// - all records    → a merged record (fields concatenated — the spec requires
///   the component field names be distinct).
///
/// Anything else — an empty list, a `%deferred` component, mixed shape classes, or
/// a higher-rank array — yields `%deferred` (a sound "don't know", never a guess).
/// The outcome of a `cat`-shape composition (spec §06/§07 `cat`): the rule
/// shared by `cat`, positional `cartprod`, positional `joint`, `joint_likelihood`
/// obstypes, and joint variates. Returning a classification (rather than a bare
/// `Deferred`) lets a caller distinguish a genuine *mixing-shape-classes static
/// error* from a merely *not-yet-resolved* component — so it can raise a precise
/// diagnostic without firing on partial inference.
enum CatShape {
    /// A well-formed cat type (all components share a shape class).
    Cat(Type),
    /// Genuinely different recognized shape classes (a static error, spec §06).
    Mixed,
    /// A component is deferred or an unclassifiable shape (higher-rank array,
    /// exotic type) — defer quietly, no error.
    Unresolved,
}

/// The single `cat` shape rule (spec §06/§07): all-scalar components → a
/// length-n vector; all-1-D-vector components → one concatenated vector (static
/// total, or dynamic if any input is); all-record components → a merged record.
/// Element types are unified by `promote2`. See [`CatShape`] for the outcomes.
fn cat_shape(types: &[Type]) -> CatShape {
    if types.is_empty() || types.iter().any(|t| matches!(t, Type::Deferred)) {
        return CatShape::Unresolved;
    }
    // Classify each component: 0 = scalar, 1 = 1-D vector, 2 = record; `None` for
    // anything else (higher-rank array, tuple, measure, …).
    let class = |t: &Type| match t {
        Type::Scalar(_) => Some(0u8),
        Type::Array { shape, .. } if shape.len() == 1 => Some(1),
        Type::Record(_) => Some(2),
        _ => None,
    };
    let classes: Vec<Option<u8>> = types.iter().map(class).collect();
    if classes.iter().any(Option::is_none) {
        return CatShape::Unresolved; // an unclassifiable component — defer, don't error
    }
    let first = classes[0].unwrap();
    if classes.iter().any(|c| c.unwrap() != first) {
        return CatShape::Mixed;
    }
    // Unify the element type across a uniform shape class.
    let promote_elems = |elems: &[&Type]| -> Option<Type> {
        let mut acc: Option<Type> = None;
        for e in elems {
            acc = Some(match acc {
                None => (*e).clone(),
                Some(prev) if &prev == *e => prev,
                Some(prev) => match promote2(Some(&prev), Some(e)) {
                    Type::Deferred => return None, // elements don't unify
                    p => p,
                },
            });
        }
        Some(acc.unwrap_or(Type::Any))
    };
    match first {
        // all scalars → a length-n vector
        0 => {
            let elems: Vec<&Type> = types.iter().collect();
            match promote_elems(&elems) {
                Some(elem) => CatShape::Cat(Type::Array {
                    shape: Box::new([Dim::Static(types.len() as u32)]),
                    elem: Box::new(elem),
                }),
                None => CatShape::Unresolved,
            }
        }
        // all 1-D vectors → one concatenated vector (static total / dynamic)
        1 => {
            let mut total = 0u32;
            let mut dynamic = false;
            let mut elems: Vec<&Type> = Vec::with_capacity(types.len());
            for t in types {
                let Type::Array { shape, elem } = t else {
                    unreachable!("class 1 is a 1-D array")
                };
                match shape[0] {
                    Dim::Static(n) => total += n,
                    Dim::Dynamic => dynamic = true,
                }
                elems.push(elem.as_ref());
            }
            match promote_elems(&elems) {
                Some(elem) => CatShape::Cat(Type::Array {
                    shape: Box::new([if dynamic {
                        Dim::Dynamic
                    } else {
                        Dim::Static(total)
                    }]),
                    elem: Box::new(elem),
                }),
                None => CatShape::Unresolved,
            }
        }
        // all records → a merged record (component fields assumed distinct)
        _ => {
            let mut fields: Vec<(Symbol, Type)> = Vec::new();
            for t in types {
                let Type::Record(fs) = t else {
                    unreachable!("class 2 is a record")
                };
                fields.extend(fs.iter().cloned());
            }
            CatShape::Cat(Type::Record(fields.into()))
        }
    }
}

/// The `cat` type for callers that only want the type (a deferral covers both
/// "mixed" and "unresolved"); callers that diagnose mixing use [`cat_shape`].
fn cat_compose(types: &[Type]) -> Type {
    match cat_shape(types) {
        CatShape::Cat(t) => t,
        CatShape::Mixed | CatShape::Unresolved => Type::Deferred,
    }
}

/// Emit the spec-§06/§07 "components must share a shape class" diagnostic at
/// `anchor` when `types` is a genuine mixing (not a partial-inference deferral),
/// and return the cat type. Shared by the `cat` op, positional `cartprod`, and
/// positional `joint`.
fn cat_or_diagnose(
    inf: &mut Inferencer<'_, '_>,
    anchor: NodeId,
    what: &str,
    types: &[Type],
) -> Type {
    match cat_shape(types) {
        CatShape::Cat(t) => t,
        CatShape::Mixed => {
            inf.diags.push(crate::Diagnostic::error_at(
                anchor,
                format!(
                    "{what} components must share a shape class (spec §06): mixing scalars, \
                     vectors, and records is a static error"
                ),
            ));
            Type::Deferred
        }
        CatShape::Unresolved => Type::Deferred,
    }
}

/// The input names of a callee that is a LOCAL measure-algebra expression typed
/// as a kernel — a fan-out `joint(K1, K2, …)`, a nested one, `truncate(K)` — where
/// the parameter list lives in the type because there is no boundary node to read.
/// `None` for anything else, so [`user_arity_check`] stays silent rather than
/// guessing.
///
/// Deliberately gated on the callee resolving, through self-module refs, to a
/// LOCAL builtin call with no `inputs` boundary. That excludes the two sources
/// whose declared list this function has no business policing:
///
/// - a cross-module `(%ref <alias> member)`, whose input list rode a side-table
///   from the dependency's own inference rather than being declared here — the
///   case the doc comment's "declares no parameter list here" already excepted;
/// - a reification, which [`user_arity_check`] reads from the boundary first and
///   never reaches this fallback for.
///
/// **An EMPTY list is a don't-know sentinel, never a declaration, and is
/// declined.** §04 forbids the thing it would otherwise mean: "No callables may
/// have nullary inputs, as this would make them equivalent to known values." So a
/// kernel type carrying zero inputs cannot be a well-formed declaration, and at
/// least one rule uses the empty list to say exactly that — `disintegrate_type`
/// documents "Falls back to empty kernel inputs … when the joint isn't a record
/// measure or the selector isn't a static field-name set", and the element reaches
/// here through a `get`, which IS a local builtin call with no boundary. Reading
/// that sentinel as `want = 0` blamed the call site for the arity of a kernel
/// whose inputs the engine could not determine, turning a program that inferred
/// and deferred honestly into a hard static error.
///
/// Declining the empty list is preferred over allowlisting heads the way
/// [`kernel_mass_is_own_rules`] does, for two reasons. It rests on a spec sentence
/// rather than on a list that has to be maintained, and it keeps the §04 check
/// reachable for every op that lifts pointwise to a kernel — an allowlist would
/// silently stop checking the next one added. It does not defend against a future
/// rule inventing a DIFFERENT sentinel (a wrong non-empty list); nothing does that
/// today.
fn local_kernel_inputs(
    inf: &Inferencer<'_, '_>,
    callee: NodeId,
    callee_ty: &Type,
) -> Option<Vec<Symbol>> {
    let Type::Kernel { inputs, .. } = callee_ty else {
        return None;
    };
    if inputs.is_empty() {
        return None;
    }
    let mut node = callee;
    loop {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(c) if c.inputs.is_none() => {
                let CallHead::Builtin(_) = c.head else {
                    return None;
                };
                return Some(inputs.to_vec());
            }
            _ => return None,
        }
    }
}

/// Reject an application of a user-defined callable whose argument count
/// contradicts the callable's declared parameter list. Returns
/// `Some(Type::Failed)` for a mismatch, `None` when the count matches, the
/// count is not knowable (see [`supplied_arg_count`]), or the callee declares no
/// parameter list this function can read (a cross-module callee's list is
/// carried in a side-table, not declared here).
///
/// Without this the call still types — `substituted_result` binds parameters to
/// arguments by keyword then position and ignores the rest — so a wrong-arity
/// application reaches the determiniser as a `ResidualUserCall` refusal instead
/// of a static error naming the callee.
///
/// **Two sources for the parameter list.** A reification declares its inputs on
/// the node ([`local_reification`] + [`input_entries`]). A callable built by a
/// measure-algebra op — a fan-out kernel `joint(K1, K2, …)`, and by the same
/// token `truncate(K)` or a nested `joint` — declares them in its TYPE instead,
/// because there is no boundary to read ([`local_kernel_inputs`]). §04 "Calling
/// conventions" governs both ("A call with field or column names that do not
/// match the callable's argument names is a static error"), so reading only the
/// first source abandoned the whole check for the second: `KJ()`,
/// `KJ(nope = 0.0)` and `KJ(z = 0.0, extra = 1.0)` all typed as a closed
/// `%measure`, and `draw(KJ())` as a concrete `%record`, with the declared input
/// never bound.
fn user_arity_check(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    callee: NodeId,
    callee_ty: &Type,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    // A §09 standard-module application is checked against its catalogue row,
    // not against a local reification's inputs.
    if inf.module_catalogue_ref(callee).is_some() {
        return None;
    }
    let declared: Vec<Symbol> = match local_reification(inf, callee) {
        Some((reif_id, _)) => input_entries(inf, reif_id)?
            .into_iter()
            .map(|(name, _)| name)
            .collect(),
        None => local_kernel_inputs(inf, callee, callee_ty)?,
    };
    let want = declared.len();
    // §04 scopes auto-splatting to "built-in or user defined value functions,
    // constructors or transition kernels", so a user call reads a sole record
    // argument exactly as a builtin call does. #78's single-input carve-out turns
    // on a DOCUMENTED DOMAIN, which a user callable does not have — its boundary
    // declares parameters, not domains — so a user call is never exempt.
    let reading = arg_reading(args, named, false)?;
    let got = reading.count;
    let who = match inf.module.node(callee) {
        Node::Ref(r) if r.ns == RefNs::SelfMod => {
            format!("`{}`", inf.module.resolve(r.name))
        }
        _ => "callable".to_string(),
    };
    if got == want {
        // The count is right; the names still have to be the declared inputs.
        let names: Vec<String> = declared
            .iter()
            .map(|n| inf.module.resolve(*n).to_string())
            .collect();
        return arg_name_check(inf, &names, &who, None, &reading, args, named);
    }
    let noun = if want == 1 { "parameter" } else { "parameters" };
    // Same reason as `arity_check`'s: a splatting call reports the field count, not
    // the one argument written. This is the path the transport-model spelling hits.
    let hint = if reading.splatting { SPLAT_HINT } else { "" };
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!("{who} declares {want} {noun}, got {got} arguments{hint}"),
    ));
    Some(Type::Failed(
        format!("user call declares {want} {noun}, got {got}").into(),
    ))
}

/// Calling a user-defined callable: a function returns its body's type, a
/// kernel returns the *measure* its body denotes (`kernelof` reifies the law
/// of a value-typed body).
fn user_call_type(
    inf: &mut Inferencer<'_, '_>,
    callee: NodeId,
    callee_ty: &Type,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    // §09 standard-module application (`hepphys.CrystalBall(args)` /
    // `specfns.erf(x)`): the callee is a `RefNs::Module` ref whose catalogue
    // signature was stashed at resolution time. Lower it with the concrete
    // call args — a Distribution sig yields the measure with the catalogue
    // `MassTag`-derived mass (Normalized or Finite) already concrete, a
    // Function sig yields the result type. This is the §09 analogue of the
    // base-distribution / per-name-function dispatch in the builtin-call path,
    // which the user-call path bypasses.
    // Surface the honest-degrade note (spec policy): when this catalogue
    // row's support/shape is a sound approximation of the spec entry that
    // the type system cannot express exactly, the user sees why.
    if let Some(cref) = inf.module_catalogue_ref(callee) {
        let note = cref.degraded.clone(); // clone to drop the borrow before &mut inf call
        if let Some(note) = note {
            inf.note_once_str(&note);
        }
        return catalogue_call_type(inf, callee, args);
    }
    // Prefer the per-call substituted body type (arg types bound to the
    // callable's parameters); fall back to the un-substituted body type for
    // cross-module callables and any case substitution can't bind.
    match callee_ty {
        Type::Function { .. } => substituted_result(inf, callee, args, named)
            .map(|(ty, _)| ty)
            .or_else(|| reified_result_type(inf, callee))
            .unwrap_or(Type::Deferred),
        // A fan-out kernel (`joint(K1, K2, …)`) is not a reification, so neither
        // substitution nor a reified body reaches its output measure —
        // `kernel_joint_result_type` builds it from the components (spec §06: "At
        // each input point the result is the `joint` of the component output
        // measures").
        // A LIFT kernel (`ksuperpose(K, w)`) is not a reification either, and it
        // is tried FIRST: the family-axis rule it enforces
        // ([`ksuperpose_family_check`]) must run even when the component happens
        // to be a reification that `substituted_result` could type on its own.
        Type::Kernel { mass, .. } => match ksuperpose_result_type(inf, callee, args, named)
            .or_else(|| substituted_result(inf, callee, args, named).map(|(ty, _)| ty))
            .or_else(|| reified_result_type(inf, callee))
            .or_else(|| kernel_joint_result_type(inf, callee))
        {
            Some(Type::Measure { domain, .. }) => Type::Measure {
                domain,
                mass: *mass,
            },
            // A rejected application (the `ksuperpose` family-axis rule) stays
            // failed; wrapping it would publish a measure over `(%failed …)`.
            Some(failed @ Type::Failed(_)) => failed,
            Some(value_ty) => Type::Measure {
                domain: Box::new(value_ty),
                mass: *mass,
            },
            None => Type::Deferred,
        },
        _ => Type::Deferred,
    }
}

/// Apply a §09 standard-module reference resolved against the catalogue. The
/// catalogue sig (stashed at resolution time, keyed by `callee`) is lowered
/// with a `LowerCtx` built from the concrete call args:
///   - Distribution sig → the lowered `Measure` with the mass from the
///     catalogue `MassTag` preserved (`Normalized` for a probability
///     distribution, `Finite` for a non-probability one such as
///     `ContinuedPoisson`). `fill_mass` leaves a concrete (non-`Deferred`)
///     mass untouched, so this rides through unchanged.
///   - Function sig → the lowered result type (scalar following the arg kind,
///     or a dynamic-dim matrix).
///
/// The support is carried separately by `catalogue_call_valueset`.
fn catalogue_call_type(inf: &mut Inferencer<'_, '_>, callee: NodeId, args: &[ArgInfo]) -> Type {
    // Clone the sig out to drop the immutable borrow before `lower`'s closures
    // re-borrow `args` inside the `&mut inf` call frame.
    let Some(sig) = inf.module_catalogue_ref(callee).map(|c| c.sig.clone()) else {
        return Type::Deferred;
    };
    // Both Distribution and Function sigs return the lowered type directly:
    // `catalogue_lower` already embeds the catalogue `MassTag` mass in the
    // `Measure` for distributions, and the result scalar/matrix type for
    // functions. No per-variant fixup is needed.
    let (ty, _vset) = catalogue_lower(&mut *inf.module, &sig, args);
    ty
}

/// The value set of an applied §09 standard-module reference: a distribution's
/// support (the lowered support ValueSet) or a function result's natural set.
/// Mirrors `distribution_support` / `function_result`'s value-set handling but
/// reads the sig from the catalogue-ref side-table rather than by op name.
fn catalogue_call_valueset(
    inf: &mut Inferencer<'_, '_>,
    callee: NodeId,
    args: &[ArgInfo],
) -> ValueSet {
    let Some(sig) = inf.module_catalogue_ref(callee).map(|c| c.sig.clone()) else {
        return ValueSet::Unknown;
    };
    catalogue_lower(&mut *inf.module, &sig, args).1
}

/// Lower a §09 catalogue sig with a `LowerCtx` built from the concrete
/// positional call args: `arg_scalar`/`arg_dim` read arg `i`'s inferred type,
/// `param_dim` (VectorFromParam) has no named-kwarg source at a `RefNs::Module`
/// application, so it falls back to the first positional arg's vector dim.
/// The `LowerCtx` borrows local closures, so it is built and consumed here in
/// one scope rather than returned.
fn catalogue_lower(
    module: &mut flatppl_core::Module,
    sig: &crate::catalogue::Sig,
    args: &[ArgInfo],
) -> (Type, ValueSet) {
    use crate::catalogue::{LowerCtx, lower};
    use std::cell::RefCell;

    let first_dim = || match args.first().map(|(_, t, _)| t) {
        Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
        _ => Dim::Dynamic,
    };
    // RefCell so the `intern` closure is a `Fn` alongside the immutable arg
    // accessors — module functions can return records (lu/svd/eigen) whose
    // field names must be interned into the current module.
    let module = RefCell::new(module);
    let ctx = LowerCtx {
        arg_scalar: &|i| match arg_ty(args, i) {
            Some(Type::Scalar(s)) => Some(*s),
            _ => None,
        },
        param_dim: &|_| first_dim(),
        arg_dim: &|i| match arg_ty(args, i) {
            Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
            _ => Dim::Dynamic,
        },
        arg_type: &|i| arg_ty(args, i).cloned(),
        intern: &|s| module.borrow_mut().intern(s),
    };
    lower(sig, &ctx)
}

/// Look through a callable expression (a `%ref` to a binding, or an inline
/// reification) to its reified body node.
fn reified_body(inf: &Inferencer<'_, '_>, mut node: NodeId) -> Option<NodeId> {
    // Deref self-module refs to the bound RHS (already inferred — typing the
    // callee forced the binding).
    loop {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == flatppl_core::RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(c) if c.inputs.is_some() => return c.args.first().copied(),
            _ => return None,
        }
    }
}

/// The codomain of `f` in `pushfwd(f, M)` (spec §06). `f` maps a value drawn from
/// `M`, so its single input is bound to `M`'s variate — type = `M`'s domain,
/// value-set = `M`'s support (read by `substituted_result` from the `M` node) —
/// and `f`'s re-inferred body type is the codomain. Falls back to `f`'s
/// un-substituted body type, then `None` (caller uses `%any`) when `f` is not a
/// resolvable reification or its body is `%deferred`.
fn pushfwd_codomain(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let f_node = args.first()?.0;
    let seed = match args.get(1) {
        Some((m_node, Type::Measure { domain, .. }, m_phase))
            if !matches!(domain.as_ref(), Type::Deferred) =>
        {
            Some(vec![(*m_node, (**domain).clone(), *m_phase)])
        }
        _ => None,
    };
    let sub = seed.and_then(|s| substituted_result(inf, f_node, &s, &[]).map(|(t, _)| t));
    match sub.or_else(|| reified_result_type(inf, f_node)) {
        Some(Type::Deferred) | None => None,
        Some(t) => Some(t),
    }
}

/// The inferred type of a callable's reified body. For a cross-module callable
/// reference the body lives in the dependency's interner and is unreachable by
/// node here, so its result type was carried over at resolution time and is
/// read from the importer's side-table; for a local callable the body is found
/// by node and looked up in the trace.
fn reified_result_type(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Option<Type> {
    if let Some(result) = inf.module_callable_result(node) {
        return Some(result.clone());
    }
    let body = reified_body(inf, node)?;
    inf.lookup_type(body).cloned()
}

/// The output variate (value domain) of a measure-algebra chain component
/// (spec §06): a base measure contributes its own `domain`; a kernel contributes
/// its reified body's output value — a `kernelof` body IS the random value (its
/// type is the variate), a `functionof`-of-measure body exposes a measure whose
/// `domain` is the variate. `None` (→ caller leaves the chain domain `%deferred`)
/// when the component is neither a measure nor a kernel, or its body / domain is
/// not statically resolvable.
fn component_variate(inf: &mut Inferencer<'_, '_>, node: NodeId, ty: &Type) -> Option<Type> {
    match ty {
        Type::Measure { domain, .. } => match domain.as_ref() {
            Type::Deferred => None,
            d => Some(d.clone()),
        },
        Type::Kernel { .. } => match reified_result_type(inf, node)? {
            Type::Measure { domain, .. } => match *domain {
                Type::Deferred => None,
                d => Some(d),
            },
            Type::Deferred => None,
            value_ty => Some(value_ty),
        },
        _ => None,
    }
}

/// `jointchain` output variate (spec §06): the `cat` of every component's
/// variate (positional form, as with `joint`), or a record naming each
/// component's variate (keyword form `jointchain(n1 = …, n2 = …)`). Any
/// component whose variate is not statically resolvable ⇒ `%deferred`.
fn jointchain_domain(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo], named: &[NamedInfo]) -> Type {
    if !named.is_empty() {
        // Keyword form `jointchain(n1 = M, n2 = K, …)`: each component's variate
        // is nested under the supplied keyword name, producing
        // `record{n1: variate(M), n2: variate(K), …}`. This nesting shape
        // (`record{name: component-variate}`) is a defensible reading of spec §06's
        // keyword form (which is defined via `relabel`, not modeled in inference).
        // Pending spec clarification on whether the keyword names wrap or replace
        // the component variate's own field names.
        let mut fields = Vec::with_capacity(named.len());
        for (name, node, t, _) in named {
            match component_variate(inf, *node, t) {
                Some(v) => fields.push((*name, v)),
                None => return Type::Deferred,
            }
        }
        return Type::Record(fields.into());
    }
    let mut variates = Vec::with_capacity(args.len());
    for (n, t, _) in args {
        match component_variate(inf, *n, t) {
            Some(v) => variates.push(v),
            None => return Type::Deferred,
        }
    }
    cat_compose(&variates)
}

/// Per-call result type of a **local** reified callable, computed by substituting
/// the concrete call-arg annotations for the callable's input parameters and
/// re-inferring its body in a throwaway module clone. This is the single-module
/// analogue of the cross-module substitution path (`modules::seed_plan` +
/// `infer_dep`): there the dependency's input *bindings* are seeded; here the
/// body's `%local` placeholder refs (or a self-bound input binding's RHS) are
/// seeded, and the body type is read back.
///
/// Without this, a callable written `f(a, b, x) = a + b * x` lowers to a
/// reification whose parameters are unconstrained `%local` placeholders
/// (`Type::Any`), so its body — and every application of it, direct or under
/// `broadcast` — types as `any`. Substituting the call's arg types makes
/// `f(1.0, 2.0, 3.0)` a `real` and `broadcast(f, x = real[5])` a `real[5]`.
///
/// Returns the substituted body's `(type, value-set)` — so a callable whose body
/// tightens its range (`f(x) = sqrt(x)` → `nonnegreals`) carries that set to the
/// call site too. `None` when `callee` is not a local reification, or no
/// parameter could be bound to an argument (caller falls back to the
/// un-substituted body).
fn substituted_result(
    inf: &mut Inferencer<'_, '_>,
    callee: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<(Type, ValueSet)> {
    let (reif_id, body) = local_reification(inf, callee)?;
    let inputs = input_entries(inf, reif_id)?;

    // Seed targets: for each parameter bound to a call argument, the body nodes
    // that read it (every matching `%local` placeholder ref, or a self-bound
    // input binding's RHS) annotated with the argument's type/phase/value-set.
    let mut seeds: Vec<(NodeId, crate::modules::Resolved)> = Vec::new();
    for (i, (sym, decl)) in inputs.iter().enumerate() {
        // Bind by keyword first (broadcast / named application), then by position.
        let arg = named
            .iter()
            .find(|(n, ..)| n == sym)
            .map(|(_, node, t, p)| (*node, t.clone(), *p))
            .or_else(|| args.get(i).map(|(node, t, p)| (*node, t.clone(), *p)));
        let Some((arg_node, ty, phase)) = arg else {
            continue;
        };
        let res = crate::modules::Resolved {
            ty,
            phase,
            vset: inf.lookup_valueset(arg_node),
            result: None,
            catalogue: None,
        };
        match decl.ns {
            RefNs::Local => collect_local_ref_seeds(inf, body, decl.name, &res, &mut seeds),
            RefNs::SelfMod => {
                if let Some(b) = inf.module.binding_by_name(decl.name) {
                    seeds.push((inf.module.binding(b).rhs, res));
                }
            }
            RefNs::Module(_) => {}
        }
    }
    if seeds.is_empty() {
        return None;
    }

    // Re-infer ONLY the body in an isolated clone seeded with the substitutions.
    // Inferring the body alone (not the whole module via `run`) avoids re-entering
    // the application that triggered this — the seeds cut every parameter, so the
    // body walk never reaches back to the call site.
    let mut clone = inf.module.clone();
    let mut sub = Inferencer::new_seeded(&mut clone, inf.level, inf.session, &seeds);
    let (ty, _) = sub.infer_node(body);
    let vset = sub.lookup_valueset(body);
    Some((ty, vset))
}

/// Deref a callee expression to its local reification: follow `self` refs to the
/// bound RHS, returning `(reification_node, body_node)` when the RHS is a
/// reification (a call carrying an inputs boundary). `None` otherwise.
fn local_reification(inf: &Inferencer<'_, '_>, mut node: NodeId) -> Option<(NodeId, NodeId)> {
    loop {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(c) if c.inputs.is_some() => return Some((node, *c.args.first()?)),
            _ => return None,
        }
    }
}

/// The ordered input parameters of a reification: `(param-name, declaration-ref)`
/// pairs. A `%specinputs` boundary carries them inline; an `%autoinputs`
/// (keyword-only) boundary reads them from the auto-inputs side-table.
fn input_entries(inf: &Inferencer<'_, '_>, reif_id: NodeId) -> Option<Vec<(Symbol, Ref)>> {
    let Node::Call(call) = inf.module.node(reif_id) else {
        return None;
    };
    match call.inputs.as_ref()? {
        Inputs::Spec(entries) => Some(entries.to_vec()),
        Inputs::Auto => inf.module.auto_inputs_of(reif_id).map(<[_]>::to_vec),
    }
}

/// Collect seeds for every `%local` placeholder ref in `body` whose name matches
/// `param`, annotating each with `res`. The body reads a parameter through these
/// placeholder refs, so seeding each makes the substituted annotation authoritative.
fn collect_local_ref_seeds(
    inf: &Inferencer<'_, '_>,
    body: NodeId,
    param: Symbol,
    res: &crate::modules::Resolved,
    out: &mut Vec<(NodeId, crate::modules::Resolved)>,
) {
    let mut stack = vec![body];
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match inf.module.node(id) {
            Node::Ref(r) if r.ns == RefNs::Local && r.name == param => {
                out.push((id, res.clone()));
            }
            Node::Call(c) => {
                if let CallHead::User(callee) = c.head {
                    stack.push(callee);
                }
                stack.extend(c.args.iter().copied());
                stack.extend(c.named.iter().map(|n| n.value));
            }
            _ => {}
        }
    }
}

/// `broadcast(f_or_K, args…)` (spec §04 broadcasting): a deterministic head
/// maps elementwise over same-shape arrays (scalars ride along) into an
/// array; a kernel / distribution-constructor head yields a **measure over
/// the array** of per-cell variates — that is why `draw` of a broadcast
/// distribution produces the observation array.
fn broadcast_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo], named: &[NamedInfo]) -> Type {
    let Some((head_node, head_ty, _)) = args.first() else {
        return Type::Deferred;
    };
    let (head_node, head_ty) = (*head_node, head_ty.clone());

    // Common shape over every data input — positional and keyword alike;
    // mismatching array shapes are deferred until real shape-broadcasting.
    let mut shape: Option<Box<[Dim]>> = None;
    let mut elems: Vec<Type> = Vec::new();
    let data_types = args[1..]
        .iter()
        .map(|(_, t, _)| t)
        .chain(named.iter().map(|(_, _, t, _)| t));
    for t in data_types {
        match t {
            Type::Array { shape: s, elem } => {
                match &shape {
                    None => shape = Some(s.clone()),
                    Some(prev) if prev == s => {}
                    Some(_) => return Type::Deferred,
                }
                elems.push(elem.as_ref().clone());
            }
            other => elems.push(other.clone()),
        }
    }
    let Some(shape) = shape else {
        // No concrete array argument. Broadcasting a DISTRIBUTION is still a
        // measure (an independent product over the eventual broadcast shape), so
        // a reified lambda body `dist.(params)` — whose params are still `%local`
        // placeholders, hence no array yet — must classify as a MEASURE so the
        // reification becomes a kernel rather than a plain function; the
        // call-site substitution then refines the concrete shape. A deterministic
        // op or a user-callable head with no array input stays a shape-wise
        // no-op (`%deferred`).
        return broadcast_distribution_no_shape(inf, head_node);
    };

    // User-callable head (`broadcast(predict, x = x_data)`): the cell comes from
    // the reified body applied to the PER-CELL argument types — an array input
    // contributes its element type, a scalar rides along — exactly the §09
    // standard-module head treatment below, and the substituted analogue of a
    // direct call.
    let cell_arg = |t: &Type| match t {
        Type::Array { elem, .. } => elem.as_ref().clone(),
        other => other.clone(),
    };
    match &head_ty {
        Type::Function { .. } => {
            let cell_args: Vec<ArgInfo> = args[1..]
                .iter()
                .map(|(n, t, p)| (*n, cell_arg(t), *p))
                .collect();
            let cell_named: Vec<NamedInfo> = named
                .iter()
                .map(|(s, n, t, p)| (*s, *n, cell_arg(t), *p))
                .collect();
            let cell = substituted_result(inf, head_node, &cell_args, &cell_named)
                .map(|(ty, _)| ty)
                .or_else(|| reified_result_type(inf, head_node))
                .unwrap_or(Type::Deferred);
            return Type::Array {
                shape,
                elem: Box::new(cell),
            };
        }
        Type::Kernel { mass, .. } => {
            let mass = *mass;
            let cell_args: Vec<ArgInfo> = args[1..]
                .iter()
                .map(|(n, t, p)| (*n, cell_arg(t), *p))
                .collect();
            let cell_named: Vec<NamedInfo> = named
                .iter()
                .map(|(s, n, t, p)| (*s, *n, cell_arg(t), *p))
                .collect();
            let cell = match substituted_result(inf, head_node, &cell_args, &cell_named)
                .map(|(ty, _)| ty)
                .or_else(|| reified_result_type(inf, head_node))
            {
                Some(Type::Measure { domain, .. }) => *domain,
                Some(value_ty) => value_ty,
                None => return Type::Deferred,
            };
            return Type::Measure {
                domain: Box::new(Type::Array {
                    shape,
                    elem: Box::new(cell),
                }),
                mass: broadcast_mass(mass),
            };
        }
        _ => {}
    }

    // §09 standard-module head (`hepphys.ContinuedPoisson`, `hepphys.interp_*`):
    // broadcast against the catalogue sig, exactly like a built-in head below.
    // Checked first because the catalogue-ref side-table is populated only for
    // §09 module references, so this never shadows a built-in. The per-cell
    // argument types feed the lowering: an array input contributes its element
    // type, a scalar rides along unchanged (every current §09 sig has a fixed
    // cell type, but lowering against the cell args keeps a future
    // `RealOrComplexOfArg` / `DomainMap` row correct).
    if let Some(sig) = inf.module_catalogue_ref(head_node).map(|c| c.sig.clone()) {
        let cell_args: Vec<ArgInfo> = args[1..]
            .iter()
            .map(|(n, t, p)| {
                let cell = match t {
                    Type::Array { elem, .. } => elem.as_ref().clone(),
                    other => other.clone(),
                };
                (*n, cell, *p)
            })
            .collect();
        return match catalogue_lower(&mut *inf.module, &sig, &cell_args).0 {
            // Distribution head: an independent product over the array. Its cell
            // domain and mass come from the catalogue sig, so a non-probability
            // measure like `ContinuedPoisson` stays `Finite`, not forced to
            // `Normalized` (mirrors the built-in distribution path below).
            Type::Measure { domain, mass } => Type::Measure {
                domain: Box::new(Type::Array {
                    shape,
                    elem: domain,
                }),
                mass: broadcast_mass(mass),
            },
            // Deterministic function head (`hepphys.interp_poly6_exp`, …): maps
            // elementwise into an array of the per-cell result, exactly as the
            // built-in deterministic-op path does.
            cell => Type::Array {
                shape,
                elem: Box::new(cell),
            },
        };
    }

    // Built-in head: a distribution constructor broadcasts into a measure
    // over the array; a deterministic scalar op maps elementwise.
    let Node::Const(op) = inf.module.node(head_node) else {
        return Type::Deferred;
    };
    let op_name = inf.module.resolve(*op).to_string();
    if let Some(cell_domain) = distribution_domain(inf, &op_name, &[], &[]) {
        return Type::Measure {
            domain: Box::new(Type::Array {
                shape,
                elem: Box::new(cell_domain),
            }),
            // Independent product of per-cell distributions.
            mass: Mass::Normalized,
        };
    }

    let cell = match (op_name.as_str(), elems.as_slice()) {
        ("add" | "sub" | "mul" | "divide" | "pow" | "min" | "max", [a, b]) => {
            promote2(Some(a), Some(b))
        }
        ("neg", [a]) => a.clone(),
        (
            "exp" | "log" | "sqrt" | "invlogit" | "logit" | "log1p" | "expm1" | "abs" | "sin"
            | "cos" | "tan" | "tanh",
            [a],
        ) => real_or_complex(Some(a)),
        // The comparisons and the logical connectives, whose per-cell result is a
        // boolean whatever the cell kinds were (their catalogue rows all declare
        // `result: Scalar(Boolean)`). This is the ONLY route §07 gives an
        // elementwise comparison — its own Domains column is scalar-only, which
        // `refuse_array_comparison` now enforces — so `v .> 3.0`, `gt.(v, w)` and
        // `broadcast(gt, v, 3.0)` are how a mask is built, and the mask is what
        // §07 "Boolean reductions" hands `lany`/`lall`. Before this the whole
        // dotted family typed `%deferred`, which left that refusal naming a route
        // with no type.
        (
            "equal" | "unequal" | "lt" | "le" | "gt" | "ge" | "land" | "lor" | "lxor" | "lnot"
            | "in" | "isfinite" | "isinf" | "isnan" | "iszero",
            [_] | [_, _],
        ) => Type::Scalar(ScalarType::Boolean),
        _ => return Type::Deferred,
    };
    Type::Array {
        shape,
        elem: Box::new(cell),
    }
}

/// `broadcast` with no concrete array argument (e.g. a reified lambda body
/// `dist.(params)` whose params are still `%local` placeholders): a distribution
/// head is nonetheless a measure — its broadcast is an independent product over a
/// not-yet-known shape — so report `(%measure %deferred · normalized)` to flag
/// it as a measure (the reification then classifies as a kernel; the call-site
/// substitution supplies the concrete domain). A §08 builtin or §09 module
/// distribution both count; anything else (deterministic op, user callable) is a
/// shape-wise no-op and stays `%deferred`.
fn broadcast_distribution_no_shape(inf: &mut Inferencer<'_, '_>, head_node: NodeId) -> Type {
    use crate::catalogue::Sig;
    // §09 module distribution head (catalogue-ref side-table). `.map().unwrap_or`
    // drops the borrow before the `&mut inf` call below.
    let module_dist = inf
        .module_catalogue_ref(head_node)
        .map(|c| matches!(c.sig, Sig::Distribution { .. }))
        .unwrap_or(false);
    // §08 builtin distribution head: resolve the name (immutable borrow ends with
    // the owned `String`), then probe `distribution_domain` (needs `&mut inf`).
    let builtin_name = match inf.module.node(head_node) {
        Node::Const(op) => Some(inf.module.resolve(*op).to_string()),
        _ => None,
    };
    let builtin_dist = builtin_name
        .map(|n| distribution_domain(inf, &n, &[], &[]).is_some())
        .unwrap_or(false);
    if module_dist || builtin_dist {
        Type::Measure {
            domain: Box::new(Type::Deferred),
            mass: Mass::Normalized,
        }
    } else {
        Type::Deferred
    }
}

/// Reject a call whose argument count contradicts the callee's declared
/// parameter list — §07 for a function or arity-only row, §08 for a
/// distribution constructor. Returns `Some(Type::Failed)` for a mis-arity call
/// and `None` when the call is admissible, the count is not knowable
/// (see [`supplied_arg_count`]), or the catalogue declares no arity for `name`.
/// The refusal for a sole positional record or table splatted onto a row that declares no names
/// for its inputs — `cat`, `get`, `get0` today.
///
/// **Grounded PER ROW, not on §04's always-splat rule.** `wave-CATADJ-report.md` §1 adjudicates
/// the auto-splat bullet's scope over special operations as **UNDERDETERMINED**: §04 never states
/// the exclusion, §03 "Tables" applies splatting to `table(r)`/`record(t)` — both on §04's
/// special-operations list — and §05 publishes a shorter roster omitting these rows. So the
/// justification below stands on §07, which holds under either reading:
///
/// - **`cat` — §07 defines no one-argument form.** Its three bullets
///   (`cat(scalar1, scalar2, ...)`, `cat(vector1, vector2, ...)`, `cat(record1, record2, ...)`)
///   are all written with two-or-more operands, and no sentence assigns a value to `cat(r)`.
///   "Arity 1..∞" is this crate's catalogue row, not spec text: "arity" occurs once in the whole
///   spec and never with a number for `cat`. Refusing contradicts no sentence.
/// - **`get`/`get0` — §07 requires a selector.** "`get(container, selectors...)` — unified element
///   access and subset selection", with every documented form supplying one and no zero-selector
///   form defined. `get0` is the "zero-based variant of `get`" and inherits the requirement.
/// - **A table** additionally falls outside each row's §07 Domains cell (`cat`'s reads "scalars,
///   vectors, or records").
///
/// §04's mismatch clause ("A call with field or column names that do not match the callable's
/// argument names is a static error") is still what the message cites for the *rule*, because on
/// a row with no declared names nothing can match — that part is not in dispute. What the
/// adjudication removed is the claim that §04's list settles which rows are in scope.
///
/// **Deliberately does NOT append [`SPLAT_HINT`].** That tail advises the keyword spelling
/// (`f(pars = record(...))`), which is unusable here: a row declaring no input names has no
/// keyword to address them by, so pointing at one would send the author down a path that cannot
/// work. The honest fix is to pass the arguments explicitly, which is what this message says.
///
/// A row that DOES declare names never reaches here — see
/// [`crate::catalogue::Catalogue::base_has_unnamed_variadic`], which steps aside for it so the
/// ordinary name check decides. That is what makes `builtin_sample`'s matching splat valid.
fn refuse_splat_onto_unnamed_variadic(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    cat: &crate::catalogue::Catalogue,
    args: &[ArgInfo],
) -> Type {
    let section = cat.base_param_section(name);
    let fields = splatted_field_names(inf, &args[0].1);
    // §07 "Field and element access": "`r.a` ≡ `get(r, "a")`", so dot access is the concise
    // spelling for pulling one field out.
    let how = match name {
        // The whole argument list is the variadic tail: every field becomes its own argument.
        "cat" | "vector" => {
            let listed = fields
                .iter()
                .map(|f| format!("t.{f}"))
                .collect::<Vec<_>>()
                .join(", ");
            if listed.is_empty() {
                format!("pass each element as its own argument, as in `{name}(t.a, t.b)`")
            } else {
                format!(
                    "pass each one as its own argument, as in `{name}({listed})` for an aggregate `t`"
                )
            }
        }
        // A distinguished prefix plus a variadic tail: the aggregate is very likely the
        // CONTAINER, and what is missing is the selector.
        "get" | "get0" => {
            let first = fields.first().map(String::as_str).unwrap_or("a");
            format!(
                "pass the container and each selector explicitly, as in `{name}(t, \"{first}\")` \
                 for an aggregate `t` — the aggregate is this row's `container` argument, not a \
                 list of arguments"
            )
        }
        _ => format!(
            "pass each argument explicitly; `{name}`'s variadic inputs are positional, so they \
             cannot be supplied as an aggregate"
        ),
    };
    let listed = if fields.is_empty() {
        String::new()
    } else {
        format!(
            " (its {} {})",
            if fields.len() == 1 {
                "name is"
            } else {
                "names are"
            },
            fields
                .iter()
                .map(|f| format!("`{f}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!(
            "`{name}`'s variadic inputs are UNNAMED (spec {section}), so a sole positional \
             record or table{listed} has no name to bind to — spec §04 makes a call whose \
             field or column names do not match the callable's argument names a static error. \
             To fix it, {how}"
        ),
    ));
    Type::Failed(format!("{name} cannot splat an aggregate onto unnamed variadic inputs").into())
}

/// Reject a sole positional record/table splat onto a [`Catalogue::base_never_splats`]
/// row — currently `checked` alone (design PR #78, owner ruling, decisions-log
/// 2026-08-18). Unlike [`refuse_splat_onto_unnamed_variadic`], the row DOES declare
/// parameter names, so the diagnostic cannot say "no name to bind to" (a name-matched
/// splat, e.g. `checked(record(value = 1.0, condition = true))`, WOULD bind if let
/// through) — the point instead is that §07's keyword spelling owns this construct.
fn refuse_checked_splat(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    cat: &crate::catalogue::Catalogue,
    args: &[ArgInfo],
) -> Type {
    let section = cat.base_param_section(name);
    let names = cat.base_param_names(name).unwrap_or(&[]);
    let keyword_form = format!(
        "{name}({})",
        names
            .iter()
            .map(|n| format!("{n} = ..."))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let fields = splatted_field_names(inf, &args[0].1);
    let listed = if fields.is_empty() {
        String::new()
    } else {
        format!(
            " (its {} {})",
            if fields.len() == 1 {
                "name is"
            } else {
                "names are"
            },
            fields
                .iter()
                .map(|f| format!("`{f}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!(
            "`{name}` has no special-operation splat (spec {section} owns the keyword form): \
             a sole positional record or table{listed} does not bind, whether or not its \
             names match. Use the keyword form instead, as in `{keyword_form}`"
        ),
    ));
    Type::Failed(format!("{name} does not splat a sole record or table argument").into())
}

/// The field or column names of a CONFIRMED record/table type, in declaration order; empty
/// for anything else.
fn splatted_field_names(inf: &Inferencer<'_, '_>, ty: &Type) -> Vec<String> {
    let syms: &[(flatppl_core::Symbol, Type)] = match ty {
        Type::Record(fields) => fields,
        Type::Table { columns, .. } => columns,
        _ => return Vec::new(),
    };
    syms.iter()
        .map(|(n, _)| inf.module.resolve(*n).to_string())
        .collect()
}

fn arity_check(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    let cat = crate::catalogue::builtin();
    let arity = cat.base_arity(name)?;
    let reading = arg_reading(args, named, cat.base_takes_aggregate_whole(name))?;
    let got = reading.count;
    // A splat onto a row whose variadic inputs are UNNAMED can never bind, whatever the
    // count, so this precedes the arity comparison: with a fitting column count the arity
    // would pass and the nameless row would then ACCEPT, and with a non-fitting one the
    // arity message would describe a count problem instead of the real one.
    if reading.splatting && cat.base_has_unnamed_variadic(name) {
        return Some(refuse_splat_onto_unnamed_variadic(inf, id, name, cat, args));
    }
    // `checked` (design PR #78, owner ruling, decisions-log 2026-08-18): §07's
    // keyword form owns it, so a sole positional record/table never splats here
    // even when its field names match `value`/`condition` — precedes the arity
    // and name checks the same way the unnamed-variadic refusal above does, for
    // the same reason: a fitting field count would otherwise let the splat
    // through before this can object.
    if reading.splatting && cat.base_never_splats(name) {
        return Some(refuse_checked_splat(inf, id, name, cat, args));
    }
    if arity.admits(got) {
        // The count is right; the names still have to be the declared ones. `?`
        // here means "this row declares none — accept the call", not a failure to
        // propagate; see `base_param_names` for which rows are nameless and why.
        let names = cat.base_param_names(name)?.to_vec();
        let section = cat.base_param_section(name);
        let who = format!("`{name}`");
        if let Some(ty) = check_double_bound(inf, &names, &who, section, args, named) {
            return Some(ty);
        }
        return arg_name_check(inf, &names, &who, Some(section), &reading, args, named);
    }
    // Same section mapping as the name check below, so a row documented outside §07 —
    // `bijection`, `logdensityof` — cites §06 in BOTH its diagnostics rather than only one.
    let section = cat.base_param_section(name);
    let declared = arity.describe();
    // `got` is the SPLAT count on a splatting call, so the author sees a number
    // larger than the arguments they wrote — say where it came from.
    let hint = if reading.splatting { SPLAT_HINT } else { "" };
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!("`{name}` takes {declared} (spec {section}), got {got}{hint}"),
    ));
    Some(Type::Failed(
        format!("{name} takes {declared}, got {got}").into(),
    ))
}

/// The explanation appended to any diagnostic whose argument count or names came
/// from an auto-splat. The splat is the surprising step — the author wrote one
/// argument and the error talks about several, or about names they never typed — so
/// every diagnostic on a splatting call says the splat happened and names the
/// spelling that passes the aggregate as one value (§04: "Passing a record or table
/// as one ordinary argument requires the keyword spelling, as in
/// `f(pars = record(...))`").
///
/// Deliberately does NOT say "always splats". #78's single-input carve-out made
/// that false: `sum(t)` and `lengthof(t)` do not splat. The wording states what
/// happened to THIS call instead of asserting a universal rule, which is both true
/// and the more useful thing to read. It also stays quiet about the carve-out —
/// naming it would be noise on `Poisson(record(zzz = 0.5))`, where no exemption
/// could apply.
///
/// Shared by all three paths that can report on a splatting call: the two arity
/// mismatches ([`arity_check`] for builtins, [`user_arity_check`] for user
/// callables) and the name check ([`arg_name_check`]). A call that does not splat
/// gets none of it — an ordinary over-arity call has nothing to explain.
const SPLAT_HINT: &str = " — this sole positional record or table splatted into one \
                          argument per field (spec §04), so its field names bind as \
                          argument names; to pass it as one ordinary argument use \
                          the keyword spelling, as in `f(pars = record(...))`";

/// How a call's arguments read against a declared parameter count.
struct ArgReading {
    /// The argument count the reading supplies.
    count: usize,
    /// True when the call auto-splats a sole positional record or table, so that
    /// argument's FIELD names — not the keyword names — are what bind.
    splatting: bool,
}

/// The reading of a call's arguments, or `None` when no reading can be trusted.
///
/// The plain reading counts positional plus keyword arguments: every §07
/// parameter and every §08 distribution parameter is nameable (§04 "Calling
/// conventions": "All built-in ordinary callables have a defined input order and
/// accept both positional and keyword arguments"), so `checked(value_expr,
/// condition = …)` and `Normal(mu = m, sigma = s)` each supply two.
///
/// A record or table given as the call's SOLE positional argument ALWAYS
/// auto-splats, supplying one argument per field or column — §04 "Calling
/// conventions": "`f(record(a = x, b = y, ...))` and `f(table(a = x, b = y, ...))`
/// are equivalent to `f(a = x, b = y, ...)`", and, as amended by design#74, "A
/// sole positional record or table therefore always splats: whether its field or
/// column names match the callable's argument names decides only whether the call
/// is valid, never whether the splat occurs."
///
/// So the callee's parameter NAMES do not influence the reading: they cannot make a
/// sole positional record read as one ordinary value. Passing a record as one
/// argument takes the keyword spelling instead — §04: "Passing a record or table as
/// one ordinary argument requires the keyword spelling, as in
/// `f(pars = record(...))`" — which lands in the `!named.is_empty()` arm below and
/// does not splat, as do the other two non-splatting cases §04 names ("a record
/// given alongside other arguments, or bound to a parameter by keyword").
///
/// `takes_aggregate_whole` is §04's SINGLE-INPUT CARVE-OUT (flatppl-design#78,
/// pending owner review): "A callable with exactly one input whose documented
/// domain admits records or tables is exempt and receives a sole positional record
/// or table whole, so that `sum(t)` and `lengthof(t)` reduce over the table rather
/// than splatting." Without it the splat binds by name, so `sum(t)` and
/// `lengthof(t)` were valid for no table at any column count — which made §07's
/// **Table reductions** paragraph dead prose. The exempt set is
/// `Catalogue::base_takes_aggregate_whole`, keyed on the callee's own arity and
/// documented domain; a USER callable is never exempt, having no documented domain
/// to read.
///
/// Because the splat is unconditional, the field names are always the binding
/// names, so `arg_name_check` now name-checks every sole-record call — including
/// the single-parameter §08 constructors that the earlier either-reading rule left
/// unchecked (`Poisson(record(zzz = 0.5))` was silently accepted).
///
/// §04 scopes all of this to ORDINARY callables ("built-in or user defined value
/// functions, constructors or transition kernels"). Special operations have
/// "distinguished, unnamed, ordered inputs" and never splat — they are excluded
/// structurally rather than here, since [`arity_check`] returns at its
/// `base_arity` lookup for a head the catalogue declares no parameter list for,
/// which is every special operation.
///
/// When the sole argument's type is still open, the reading is not knowable.
fn arg_reading(
    args: &[ArgInfo],
    named: &[NamedInfo],
    takes_aggregate_whole: bool,
) -> Option<ArgReading> {
    let plain = args.len() + named.len();
    let plain_reading = Some(ArgReading {
        count: plain,
        splatting: false,
    });
    if !named.is_empty() || args.len() != 1 || takes_aggregate_whole {
        return plain_reading;
    }
    let fields = match &args[0].1 {
        Type::Record(fields) => fields.len(),
        Type::Table { columns, .. } => columns.len(),
        Type::Deferred | Type::Var(_) | Type::Any | Type::Failed(_) => return None,
        // Not a record or table: an ordinary sole positional argument.
        _ => return plain_reading,
    };
    Some(ArgReading {
        count: fields,
        splatting: true,
    })
}

/// The names a call binds its arguments to, each with the node to anchor a
/// diagnostic at: the keyword arguments, or — when `reading` splats — the sole
/// record or table argument's field names (anchored at that argument, the only
/// node the type carries). Empty for a purely positional call, which binds by
/// order and so has no names to check.
fn supplied_arg_names(
    inf: &Inferencer<'_, '_>,
    reading: &ArgReading,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Vec<(String, NodeId)> {
    if !reading.splatting {
        return named
            .iter()
            .map(|(n, node, _, _)| (inf.module.resolve(*n).to_string(), *node))
            .collect();
    }
    let (node, ty, _) = &args[0];
    let fields = match ty {
        Type::Record(fields) => fields,
        Type::Table { columns, .. } => columns,
        _ => return Vec::new(),
    };
    fields
        .iter()
        .map(|(n, _)| (inf.module.resolve(*n).to_string(), *node))
        .collect()
}

/// Reject a call that binds one parameter twice — positionally and by keyword, or by
/// keyword more than once.
///
/// §04 "Calling conventions" gives positional and keyword arguments each their own
/// binding rule — "Positional arguments are accepted only if the callable has ordered
/// inputs, so that the arguments can be mapped to the inputs in order" and "Arguments
/// are bound to inputs by name" — and only mixes the two forms through "All built-in
/// ordinary callables have a defined input order and accept both positional and
/// keyword arguments." A defined input order is a mapping from position to ONE input
/// each; whatever supplies a second value to that input — a positional argument
/// filling it and a keyword naming it too, or two keywords naming it — has no input
/// left for the second value to bind to. (No sentence spells out "each input is bound
/// once" directly; this is the reading the "defined input order" framing forces, not
/// an explicit spec rule — worth tightening in a future §04 revision.)
///
/// Unenforced, `atan2(1.0, y = 2.0)` and `atan2(y = 1.0, y = 2.0)` both passed
/// silently: `arity_check` counts `args.len() + named.len()` and never notices two
/// entries naming the same input, and the name check below only verifies that a
/// supplied name IS declared, never that it is supplied once. `normalize_keyword_args`
/// already detects both collisions (the `pos < slots.len() && slots[pos].is_some()`
/// guards, one per spelling) but only to hand the call back unnormalized — silently,
/// since normalization failure is not itself an error path. This makes both collisions
/// a static error instead of a mapping the normalizer quietly declines.
///
/// Only reachable on the mixed or all-keyword spelling: a splat has no separate
/// keyword list to collide within (`arg_reading` only splats when `named` is empty,
/// and a splatted aggregate's fields are unique by construction), so `args` and
/// `named` here are the ordinary positional prefix and keyword suffix of one call.
fn check_double_bound(
    inf: &mut Inferencer<'_, '_>,
    declared: &[String],
    who: &str,
    section: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    // `slots[pos]` is `true` once something has claimed that declared position — the
    // positional prefix claims its slots up front, so a keyword landing on one of them
    // is caught the same way a second keyword landing on an already-keyword-claimed one is.
    let mut slots = vec![false; declared.len()];
    for slot in slots.iter_mut().take(args.len()) {
        *slot = true;
    }
    let mut offenders: Vec<(String, NodeId, bool)> = Vec::new();
    for (sym, node, ..) in named {
        let supplied = inf.module.resolve(*sym).to_string();
        let Some(pos) = declared.iter().position(|d| *d == supplied) else {
            continue; // undeclared name — `arg_name_check`'s business, not this check's
        };
        if slots[pos] {
            // `pos < args.len()` distinguishes "a positional argument already claimed
            // this slot" from "an earlier keyword already claimed it" for the message.
            offenders.push((supplied, *node, pos < args.len()));
        } else {
            slots[pos] = true;
        }
    }
    if offenders.is_empty() {
        return None;
    }
    for (name, at, by_position) in &offenders {
        let how = if *by_position {
            "bound both positionally and by keyword"
        } else {
            "bound by keyword more than once"
        };
        inf.diags.push(crate::Diagnostic::error_at(
            *at,
            format!(
                "{who} parameter `{name}` is {how} (spec {section} parameters have a \
                 defined input order — each is bound once)"
            ),
        ));
    }
    Some(Type::Failed(
        format!("{who} parameter `{}` is bound twice", offenders[0].0).into(),
    ))
}

/// Reject a call that binds an argument to a name the callee does not declare.
///
/// §04 "Calling conventions" makes keyword arguments bind by name — "Arguments
/// are bound to inputs by name, the order of the arguments is not relevant" —
/// and states the rule outright for the auto-splat form: "A call with field or
/// column names that do not match the callable's argument names is a static
/// error." Unenforced, `Normal(mu = 0.0, tau = 1.0)` passes on count alone and
/// determinizes to `builtin_logdensityof(Normal, record(mu = 0.0, tau = 1.0), …)`
/// — a nonexistent `tau` and a missing `sigma` handed to the engine.
///
/// Reports every unbindable name, and returns `Some(Type::Failed)` if there was
/// one. `who` names the callee as it should read in the diagnostic; `section`
/// attributes the parameter list ("spec §08" for a constructor row, `None` for a
/// user callable, whose parameters are declared in the module itself).
///
/// A SPLATTING call gets the keyword spelling named too. Its field names bind
/// because §04 makes a sole positional record splat unconditionally, so the author
/// who meant to pass the record as one value has an actionable fix rather than a
/// bare "no parameter" — §04: "Passing a record or table as one ordinary argument
/// requires the keyword spelling, as in `f(pars = record(...))`". The hint goes on
/// every diagnostic, not once, because each is read on its own line in an editor.
fn arg_name_check(
    inf: &mut Inferencer<'_, '_>,
    declared: &[String],
    who: &str,
    section: Option<&str>,
    reading: &ArgReading,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    let supplied = supplied_arg_names(inf, reading, args, named);
    let unknown: Vec<(String, NodeId)> = supplied
        .into_iter()
        .filter(|(n, _)| !declared.iter().any(|d| d == n))
        .collect();
    if unknown.is_empty() {
        return None;
    }
    let list = declared
        .iter()
        .map(|p| format!("`{p}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let source = match section {
        Some(s) => format!("spec {s} parameters"),
        None => "declares".to_string(),
    };
    // The splat is what made these field names binding, so say so.
    let hint = if reading.splatting { SPLAT_HINT } else { "" };
    for (name, at) in &unknown {
        inf.diags.push(crate::Diagnostic::error_at(
            *at,
            format!("{who} has no parameter `{name}` ({source}: {list}){hint}"),
        ));
    }
    Some(Type::Failed(
        format!("{who} has no parameter `{}`", unknown[0].0).into(),
    ))
}

/// The result type of a per-name function declared in the catalogue as
/// `Sig::Function`, or `None` if the name is not a known function (so the
/// caller can fall through to distribution dispatch, then gap).
///
/// `arg_scalar` is built from the inferred positional argument types so that
/// `RealOrComplexOfArg` and `DomainMap` sigs can read the call-site scalar kind.
fn function_result(
    module: &mut flatppl_core::Module,
    name: &str,
    args: &[ArgInfo],
) -> Option<Type> {
    use crate::catalogue::{LowerCtx, Sig, lower};
    use std::cell::RefCell;

    let sig = crate::catalogue::builtin().base(name)?;
    // Only Function rows here; Distribution rows are handled by distribution_domain.
    let Sig::Function { .. } = sig else {
        return None;
    };
    // `ResultSig::Record` interns field-name Symbols into the module. Behind a
    // RefCell so the `intern` closure is a `Fn` (not `FnMut`), coexisting with
    // the immutable arg accessors. This is the one live function-row lower path,
    // so it gets the real interner (other sites use `no_intern`).
    let module = RefCell::new(module);
    let ctx = LowerCtx {
        arg_scalar: &|i| match arg_ty(args, i) {
            Some(Type::Scalar(s)) => Some(*s),
            _ => None,
        },
        param_dim: &|_| Dim::Dynamic,
        arg_dim: &|i| match arg_ty(args, i) {
            Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
            _ => Dim::Dynamic,
        },
        arg_type: &|i| arg_ty(args, i).cloned(),
        intern: &|s| module.borrow_mut().intern(s),
    };
    let (ty, _) = lower(sig, &ctx);
    Some(ty)
}

/// The value-set of a per-name catalogue `Sig::Function` result — its `result_set`
/// tag lowered with the concrete arg types (`sqrt → nonnegreals`, `lengthof →
/// nonnegintegers`, `Natural` rows → the type's natural extent). `None` when the
/// name is not a known function row, so the caller falls through to distribution
/// support. Mirrors `function_result`, returning the value-set arm of `lower`.
fn function_valueset(
    module: &mut flatppl_core::Module,
    name: &str,
    args: &[ArgInfo],
) -> Option<ValueSet> {
    use crate::catalogue::{LowerCtx, Sig, lower};
    use std::cell::RefCell;

    let sig = crate::catalogue::builtin().base(name)?;
    let Sig::Function { .. } = sig else {
        return None;
    };
    let module = RefCell::new(module);
    let ctx = LowerCtx {
        arg_scalar: &|i| match arg_ty(args, i) {
            Some(Type::Scalar(s)) => Some(*s),
            _ => None,
        },
        param_dim: &|_| Dim::Dynamic,
        arg_dim: &|i| match arg_ty(args, i) {
            Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
            _ => Dim::Dynamic,
        },
        arg_type: &|i| arg_ty(args, i).cloned(),
        intern: &|s| module.borrow_mut().intern(s),
    };
    let (_, vset) = lower(sig, &ctx);
    Some(vset)
}

/// The variate domain of a spec-§08 distribution constructor, or `None` when
/// the name is not a known distribution.
///
/// Dispatches via the catalogue for all 30 base distributions; non-distribution
/// names fall through to `None` unchanged.
fn distribution_domain(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    use crate::catalogue::{LowerCtx, Sig, lower};

    let sig = crate::catalogue::builtin().base(name)?;
    // Confirm it's a distribution sig (Function rows are not distributions).
    let Sig::Distribution { .. } = sig else {
        return None;
    };
    // Build a context whose `param_dim` delegates to the existing helper
    // so that VectorFromParam dims (MvNormal/Dirichlet/Multinomial) resolve
    // the same way as before.  The closure borrows `inf` as `&Inferencer`
    // (a shared reborrow of the `&mut`); it is dropped before `inf` is
    // used mutably again.
    let ctx = LowerCtx {
        param_dim: &|kwarg| param_dim(inf, args, named, kwarg),
        arg_scalar: &|_| None,
        arg_dim: &|_| Dim::Dynamic,
        arg_type: &|i| arg_ty(args, i).cloned(),
        intern: &crate::catalogue::no_intern,
    };
    let (ty, _vset) = lower(sig, &ctx);
    // `lower` wraps the domain in a `Type::Measure`; unwrap to get the domain.
    if let Type::Measure { domain, .. } = ty {
        Some(*domain)
    } else {
        None
    }
}

/// The `None`-variate fallback for `builtin_sample` / the transports: a resolved-but-
/// non-kernel `kernel` argument is a static error (§07 operates on a kernel object); a
/// still-pending type (`%deferred` / a type variable) defers, cleared by re-inference.
/// `Failed` / `Any` stay silent — the cause was reported elsewhere, or is unconstrained.
fn non_kernel_or_defer(
    inf: &mut Inferencer<'_, '_>,
    kernel: Option<&ArgInfo>,
    op: &str,
    argpos: &str,
) -> Type {
    match kernel {
        Some((kn, kt, _))
            if !matches!(
                kt,
                Type::Deferred
                    | Type::Var(_)
                    | Type::Failed(_)
                    | Type::Any
                    | Type::Kernel { .. }
                    | Type::Measure { .. }
            ) =>
        {
            inf.diags.push(crate::Diagnostic::error_at(
                *kn,
                format!(
                    "{op}: {argpos} must be a distribution kernel — a built-in \
                     constructor or a reified kernel (spec §07)"
                ),
            ));
            Type::Failed(format!("{op}: non-kernel argument").into())
        }
        _ => Type::Deferred,
    }
}

/// The variate domain of the measure `kernel(kernel_input)` would produce, for a
/// kernel given as a bare distribution constructor — a base built-in (§08) or a §09
/// module member (`hepphys.Argus`). The `distribution_domain` pattern, but the name
/// comes from the `kernel` argument node and the length params come from the
/// `kernel_input` record (not call args). `None` when the kernel is not a (base or
/// module) distribution constructor.
fn kernel_variate(
    inf: &mut Inferencer<'_, '_>,
    kernel_node: NodeId,
    kernel_input_node: Option<NodeId>,
) -> Option<Type> {
    use crate::catalogue::{LowerCtx, Sig, lower, no_intern};

    // Lower a distribution `Sig` to its variate domain. `param_dim` reads the
    // kernel_input RECORD's field `kwarg` (vs `distribution_domain`'s call args);
    // scalar / matrix dists never call it, shaped dists (`MvNormal`/…) do.
    let lower_dist = |inf: &mut Inferencer<'_, '_>, sig: &Sig| -> Option<Type> {
        let pd = |kwarg: &str| record_field_dim(inf, kernel_input_node, kwarg);
        let ctx = LowerCtx {
            param_dim: &pd,
            arg_scalar: &|_| None,
            arg_dim: &|_| Dim::Dynamic,
            arg_type: &|_| None,
            intern: &no_intern,
        };
        match lower(sig, &ctx).0 {
            Type::Measure { domain, .. } => Some(*domain),
            _ => None,
        }
    };

    // §09 module member (e.g. `hepphys.Argus`): the stashed catalogue ref carries the
    // Sig (alias→module resolved at ref-resolution time, as `catalogue_call_type` reads
    // it). Clone the Sig to drop the borrow before the `&mut inf` lower. A module ref
    // that isn't a distribution is not a kernel — `None`, not a fall-through to base.
    if let Some(sig) = inf.module_catalogue_ref(kernel_node).map(|c| c.sig.clone()) {
        return match sig {
            Sig::Distribution { .. } => lower_dist(inf, &sig),
            _ => None,
        };
    }

    // Base built-in constructor: the kernel node is a builtin head (or a bare const).
    let name = match inf.module.node(kernel_node) {
        Node::Call(c) => match c.head {
            CallHead::Builtin(op) => inf.module.resolve(op).to_string(),
            _ => return None,
        },
        Node::Const(sym) => inf.module.resolve(*sym).to_string(),
        _ => return None,
    };
    let sig = crate::catalogue::builtin().base(&name)?;
    let Sig::Distribution { .. } = sig else {
        return None;
    };
    lower_dist(inf, sig)
}

/// Leading array dim of the kernel_input record's `kwarg` field, for shaped dists
/// (`MvNormal` `mu`, `Dirichlet` `alpha`, `Multinomial` `p`). The kernel_input is a
/// `record(...)` call whose fields are its `named` args (`NamedKind::Field`); read the
/// named field's value type. `Dim::Dynamic` if the input is absent / not a record /
/// lacks the field / the field is not an array — the honest under-approximation (matrix
/// dists never call this; a not-yet-inferred field also yields `Dynamic`).
fn record_field_dim(inf: &Inferencer<'_, '_>, rec: Option<NodeId>, kwarg: &str) -> Dim {
    let Some(rec) = rec else { return Dim::Dynamic };
    let Node::Call(c) = inf.module.node(rec) else {
        return Dim::Dynamic;
    };
    if !matches!(c.head, CallHead::Builtin(op) if inf.module.resolve(op) == "record") {
        return Dim::Dynamic;
    }
    for na in c.named.iter() {
        if inf.module.resolve(na.name) == kwarg {
            return match inf.lookup_type(na.value) {
                Some(Type::Array { shape, .. }) => shape.first().copied().unwrap_or(Dim::Dynamic),
                _ => Dim::Dynamic,
            };
        }
    }
    Dim::Dynamic
}

/// The value NODE of `field` when a call auto-splats a positional record (§04
/// "Calling conventions"): auto-splat fires ONLY when a record is the call's
/// SOLE argument, so this returns `None` unless there are no keyword arguments,
/// exactly one positional argument, and that argument is a `record(...)` call
/// carrying `field`. A record alongside other arguments, or bound to a
/// parameter by keyword, is an ordinary value and is not splatted.
///
/// The §06 fundamental measures (`Dirac`/`Lebesgue`/`Counting`) resolve their
/// single argument by hand (they are not in the §08 catalogue), so they consult
/// this to splat a sole positional record the way the catalogue param path does
/// for distributions — `Dirac(record(value = v))` binds `value = v`, not the
/// whole record.
fn splat_field(
    inf: &Inferencer<'_, '_>,
    args: &[ArgInfo],
    named: &[NamedInfo],
    field: &str,
) -> Option<NodeId> {
    if !named.is_empty() || args.len() != 1 {
        return None;
    }
    let Node::Call(c) = inf.module.node(args[0].0) else {
        return None;
    };
    if !matches!(c.head, CallHead::Builtin(op) if inf.module.resolve(op) == "record") {
        return None;
    }
    c.named
        .iter()
        .find(|na| inf.module.resolve(na.name) == field)
        .map(|na| na.value)
}

/// The `support` argument NODE of a `Lebesgue`/`Counting` call (spec §06
/// "Fundamental measures" table: parameter name `support`, e.g. `Lebesgue(reals)`
/// or `Counting(integers)` — §11 mass-class worked example). Resolves the named
/// kwarg, an auto-splatted positional `record(support = S)` (§04), or the plain
/// positional set, in that order — every caller of this rule (the type-domain
/// arm and the `%mass` arm) must accept all three spellings alike.
fn lebesgue_counting_support_node(
    inf: &Inferencer<'_, '_>,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<NodeId> {
    named
        .iter()
        .find(|(n, _, _, _)| inf.module.resolve(*n) == "support")
        .map(|(_, node, _, _)| *node)
        .or_else(|| splat_field(inf, args, named, "support"))
        .or_else(|| args.first().map(|a| a.0))
}

/// A dummy `SupportTag::Structural` check helper so `distribution_support` can
/// peek at the raw tag without re-looking up the catalogue entry.
#[inline]
fn support_is_structural(sig: &crate::catalogue::Sig) -> bool {
    use crate::catalogue::{Sig, SupportTag};
    matches!(
        sig,
        Sig::Distribution {
            support: SupportTag::Structural,
            ..
        }
    )
}

/// The static dim of a distribution's length-defining parameter (`mu`,
/// `alpha`, `p`): its inferred type's single array dim, at `Level::Shape`.
fn param_dim(inf: &Inferencer<'_, '_>, args: &[ArgInfo], named: &[NamedInfo], kwarg: &str) -> Dim {
    if inf.level < Level::Shape {
        return Dim::Dynamic;
    }
    let ty = named
        .iter()
        .find(|(n, _, _, _)| inf.module.resolve(*n) == kwarg)
        .map(|(_, _, t, _)| t)
        .or_else(|| args.first().map(|(_, t, _)| t));
    match ty {
        Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
        _ => Dim::Dynamic,
    }
}

// =====================================================================
// Value sets (Level::Valueset) — the third `%meta` slot
// =====================================================================

pub(crate) fn literal_valueset(s: &Scalar) -> ValueSet {
    match s {
        Scalar::Int(n) if *n > 0 => ValueSet::PosIntegers,
        Scalar::Int(n) if *n == 0 => ValueSet::NonNegIntegers,
        Scalar::Int(_) => ValueSet::Integers,
        // A real literal is its own singleton interval.
        Scalar::Real(r) => ValueSet::Interval(*r, *r),
        Scalar::Bool(_) => ValueSet::Booleans,
        Scalar::Str(_) => ValueSet::Unknown,
    }
}

pub(crate) fn const_valueset(name: &str) -> ValueSet {
    match name {
        "pi" | "inf" => ValueSet::PosReals,
        "im" => ValueSet::Complexes,
        _ => ValueSet::Unknown,
    }
}

/// The value set of a call node: a measure node's support, a value node's
/// strongest known containing set. Conservative — `Unknown` is always sound.
pub(crate) fn call_valueset(
    inf: &mut Inferencer<'_, '_>,
    call: &Call,
    callee: Option<&(NodeId, Type)>,
    args: &[ArgInfo],
    named: &[NamedInfo],
    ty: &Type,
) -> ValueSet {
    // User-callable application: the reified body's set rides over (for a
    // kernel call, the body set IS the output measure's support). A §09
    // standard-module reference has no reified body — its support/result set
    // is lowered from the catalogue sig with the call args.
    if let Some((callee_node, _)) = callee {
        if inf.module_catalogue_ref(*callee_node).is_some() {
            return catalogue_call_valueset(inf, *callee_node, args);
        }
        // Per-call substituted body value-set (arg sets bound to the callable's
        // parameters): a callable whose body tightens its range — `f(x) =
        // sqrt(x)` — carries `nonnegreals` to the call site. Fall back to the
        // un-substituted body set when substitution binds nothing or yields no
        // finer set.
        if let Some((_, vs)) = substituted_result(inf, *callee_node, args, named) {
            if vs != ValueSet::Unknown {
                return vs;
            }
        }
        return match reified_body(inf, *callee_node) {
            Some(body) => inf.lookup_valueset(body),
            None => ValueSet::Unknown,
        };
    }
    let CallHead::Builtin(op) = call.head else {
        return ValueSet::Unknown;
    };
    // Reifications are callables, not values.
    if call.inputs.is_some() {
        return ValueSet::Unknown;
    }
    let name = inf.module.resolve(op).to_string();
    match name.as_str() {
        // A set-constructor used directly as a value binding is a PRESET (spec
        // §03): its value-set is the set it denotes. (Its TYPE is `%any` — a set
        // is not a value type — set in the `cartprod`/`interval`/… type arms.)
        "interval" | "stdsimplex" | "cartpow" | "cartprod" => set_call_valueset(inf, call),
        // Parameters / loaded sets.
        "elementof" | "external" => set_expr_valueset(inf, args.first().map(|a| a.0)),
        // `broadcast(head, data…)` over a user callable: the cell value-set is
        // the substituted body's set (per-cell arg sets bound to the head's
        // parameters), lifted into a `CartPow` over the result array. So
        // `broadcast(f, v)` with `f(x) = sqrt(x)` is `cartpow(nonnegreals, n)`.
        // Other heads (built-ins / §09 modules) fall back to the natural set.
        "broadcast" => {
            let Some((head_node, _, _)) = args.first() else {
                return ValueSet::Unknown;
            };
            let head_node = *head_node;
            let cell = |t: &Type| match t {
                Type::Array { elem, .. } => elem.as_ref().clone(),
                other => other.clone(),
            };
            let cell_args: Vec<ArgInfo> = args[1..]
                .iter()
                .map(|(n, t, p)| (*n, cell(t), *p))
                .collect();
            let cell_named: Vec<NamedInfo> = named
                .iter()
                .map(|(s, n, t, p)| (*s, *n, cell(t), *p))
                .collect();
            match (
                substituted_result(inf, head_node, &cell_args, &cell_named),
                result_array_dim(ty),
            ) {
                (Some((_, cell_vs)), Some(dim)) if cell_vs != ValueSet::Unknown => {
                    ValueSet::CartPow(Box::new(cell_vs), dim)
                }
                _ => ValueSet::Unknown,
            }
        }
        // `load_data`'s value lies in the declared `valueset` itself (spec §07:
        // "`valueset` fully determines the result's shape") — no extra row axis.
        "load_data" => {
            let vs = named_or_positional_node(inf.module, named, args, "valueset", 1);
            set_expr_valueset(inf, vs)
        }
        // Measure supports (the measure node's value set IS its support).
        // `support` resolves positionally or by keyword, matching `fill_mass`'s
        // Lebesgue/Counting arm — both read the same argument.
        "Lebesgue" | "Counting" => {
            set_expr_valueset(inf, lebesgue_counting_support_node(inf, args, named))
        }
        "lawof" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        "truncate" => {
            // Sound superset: the truncated support lies inside S.
            match set_expr_valueset(inf, args.get(1).map(|a| a.0)) {
                ValueSet::Unknown => args
                    .first()
                    .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
                set => set,
            }
        }
        // Reweighting never grows the support.
        "normalize" | "bayesupdate" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        "weighted" | "logweighted" => args
            .get(1)
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        // Drawing yields a value in the measure's support.
        "draw" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        "iid" => {
            let inner = args
                .first()
                .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n));
            match (inner, ty) {
                (ValueSet::Unknown, _) => ValueSet::Unknown,
                (inner, Type::Measure { domain, .. }) => match domain.as_ref() {
                    Type::Array { shape, .. } if shape.len() == 1 => {
                        ValueSet::CartPow(Box::new(inner), shape[0])
                    }
                    _ => ValueSet::Unknown,
                },
                _ => ValueSet::Unknown,
            }
        }
        // Normalization functions (spec §07).
        "softmax" => ValueSet::StdSimplex(vector_dim(arg_ty(args, 0))),
        "l1unit" => {
            let arg_set = args
                .first()
                .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n));
            // `v/‖v‖₁` lies on the simplex only for nonnegative `v`.
            if arg_set.subset_of(&ValueSet::CartPow(
                Box::new(ValueSet::NonNegReals),
                Dim::Dynamic,
            )) {
                ValueSet::StdSimplex(vector_dim(arg_ty(args, 0)))
            } else {
                ValueSet::Unknown
            }
        }
        // Vectors lift a common element set; heterogeneous elements widen
        // to the strongest named set containing all of them (literal reals
        // are singleton intervals, so without widening `l1unit`'s simplex
        // guard would never fire on literal weight vectors).
        "vector" => {
            let sets: Vec<ValueSet> = args
                .iter()
                .map(|(n, _, _)| inf.lookup_valueset(*n))
                .collect();
            match join_scalar_sets(&sets) {
                Some(e) => ValueSet::CartPow(Box::new(e), Dim::Static(args.len() as u32)),
                None => ValueSet::Unknown,
            }
        }
        // `checked`/`fixed` are identity (spec §03), so the wrapped value's set
        // rides through — otherwise it would be needlessly lost to `Unknown`.
        "checked" | "fixed" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        // §07 "Table reductions" give a RECORD, so the value-set must be the
        // matching `cartprod(col = …)` and not the scalar set the callee's
        // catalogue row declares. Without this arm `var(<table>)` carries
        // `nonnegreals` — a scalar set on a record-typed value, which is incoherent
        // and would mislead anything reading the set rather than the type.
        // `sum`/`mean`/`prod` are `Structural` rows with no `result_set`, so they
        // already fall through to the type's natural extent and land on a record
        // set; this arm makes `var`/`std` agree instead of short-circuiting on
        // their row. `maximum`/`minimum` are `Function` rows (catalogue
        // `ElemScalarKind`, no `result_set`) that WOULD short-circuit the same way
        // `function_valueset` computes a scalar set from `ElemScalarKind` against
        // the table argument directly, ignoring the record type — so they need the
        // arm too, unlike the Structural trio.
        // `median`/`lany`/`lall` are `Function` rows too, so they need the arm for
        // the `maximum`/`minimum` reason.
        // Mirrors [`table_reduction_type`] arm for arm.
        "sum" | "mean" | "var" | "std" | "maximum" | "minimum" | "median" | "lany" | "lall"
            if matches!(arg_ty(args, 0), Some(Type::Table { .. })) =>
        {
            table_reduction_valueset(&name, arg_ty(args, 0))
        }
        // Mirrors the `cumsum`/`cumprod` §03-promotion type arm. The catalogue's
        // `SameAsArg(0)` row would hand back the ARGUMENT's set, `cartpow(booleans,
        // n)`, against an integer-element type — a set that excludes the value
        // (`cumsum([true, true, false])` is `[1, 2, 2]`). `sum`/`prod` need no arm:
        // they are `Structural` rows with no `result_set`, so their set already
        // follows the type.
        "cumsum" | "cumprod" if bool_elem_array(arg_ty(args, 0)) => {
            ValueSet::natural_of(&cumulative_bool_type(arg_ty(args, 0)))
        }
        // Catalogue functions carry their result value-set (`result_set` tag);
        // distribution constructors carry the support column of spec §08. A
        // bare name is one or the other — try the function row first, then fall
        // through to distribution support.
        _ => {
            if let Some(vs) = function_valueset(&mut *inf.module, &name, args) {
                vs
            } else {
                distribution_support(inf, &name, args, named)
            }
        }
    }
}

/// The leading dim of a rank-1 array result, drilling through a measure wrapper
/// (a broadcast deterministic head gives an array; a kernel head gives a measure
/// over an array). `None` for any other shape.
fn result_array_dim(ty: &Type) -> Option<Dim> {
    match ty {
        Type::Array { shape, .. } if shape.len() == 1 => Some(shape[0]),
        Type::Measure { domain, .. } => result_array_dim(domain),
        _ => None,
    }
}

/// The single dim of a vector-typed argument, for simplex sizes.
fn vector_dim(ty: Option<&Type>) -> Dim {
    match ty {
        Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
        _ => Dim::Dynamic,
    }
}

/// The value-set denoted by a set-constructor CALL (`interval(...)`,
/// `stdsimplex(n)`, `cartpow(S, size)`, `cartprod(...)`). Shared by
/// `set_expr_valueset` (set-expression argument position) and `call_valueset`
/// (a set-constructor used directly as a preset binding, spec §03). `Unknown`
/// for any non-set-constructor head or unresolvable component.
fn set_call_valueset(inf: &mut Inferencer<'_, '_>, c: &Call) -> ValueSet {
    let CallHead::Builtin(op) = c.head else {
        return ValueSet::Unknown;
    };
    match inf.module.resolve(op).to_string().as_str() {
        "interval" => {
            let bound = |n: Option<&NodeId>| match n.map(|&n| inf.module.node(n).clone()) {
                Some(Node::Lit(Scalar::Real(r))) => Some(r),
                Some(Node::Lit(Scalar::Int(i))) => Some(i as f64),
                Some(Node::Const(sym)) if inf.module.resolve(sym) == "inf" => Some(f64::INFINITY),
                Some(Node::Call(neg))
                    if matches!(neg.head, CallHead::Builtin(op)
                        if inf.module.resolve(op) == "neg") =>
                {
                    bound_of(inf, neg.args.first().copied()).map(|b| -b)
                }
                _ => None,
            };
            match (bound(c.args.first()), bound(c.args.get(1))) {
                (Some(lo), Some(hi)) => ValueSet::Interval(lo, hi),
                _ => ValueSet::Unknown,
            }
        }
        "stdsimplex" => ValueSet::StdSimplex(
            c.args
                .first()
                .map_or(Dim::Dynamic, |&n| resolve_dim(inf, n)),
        ),
        "cartpow" => {
            // The size is REQUIRED (spec §03 "Cartesian power") — the type
            // arm already fails 1-arg `cartpow(S)` (`Type::Failed`, above);
            // agree here rather than defaulting the missing size to
            // `%dynamic` and synthesizing a plausible-looking value-set for
            // an ill-formed call.
            let Some(&size_arg) = c.args.get(1) else {
                return ValueSet::Unknown;
            };
            let elem = set_expr_valueset(inf, c.args.first().copied());
            if elem == ValueSet::Unknown {
                return ValueSet::Unknown;
            }
            let shape = count_dims(inf, size_arg);
            flatppl_core::ty::cartpow_over(elem, &shape)
        }
        "cartprod" => {
            // Positional → CartProd; keyword → RecordSet. Mixing is not
            // a valid set expression (spec §03 gives the two forms
            // separately); if both are present, the named fields win as
            // a record and positional args are ignored (front-end
            // should already reject the mix).
            if !c.named.is_empty() {
                let mut fields = Vec::with_capacity(c.named.len());
                for na in c.named.iter() {
                    let set = set_expr_valueset(inf, Some(na.value));
                    if set == ValueSet::Unknown {
                        return ValueSet::Unknown;
                    }
                    fields.push((na.name, set));
                }
                ValueSet::RecordSet(fields.into())
            } else {
                let mut parts = Vec::with_capacity(c.args.len());
                for &arg in c.args.iter() {
                    let set = set_expr_valueset(inf, Some(arg));
                    if set == ValueSet::Unknown {
                        return ValueSet::Unknown;
                    }
                    parts.push(set);
                }
                ValueSet::CartProd(parts.into())
            }
        }
        _ => ValueSet::Unknown,
    }
}

/// A set *expression* (an `elementof` / `truncate` / reference-measure
/// argument) read structurally into a [`ValueSet`].
fn set_expr_valueset(inf: &mut Inferencer<'_, '_>, node: Option<NodeId>) -> ValueSet {
    let Some(node) = node else {
        return ValueSet::Unknown;
    };
    match inf.module.node(node).clone() {
        Node::Const(sym) => match inf.module.resolve(sym) {
            "reals" => ValueSet::Reals,
            "posreals" => ValueSet::PosReals,
            "nonnegreals" => ValueSet::NonNegReals,
            "unitinterval" => ValueSet::UnitInterval,
            "integers" => ValueSet::Integers,
            "posintegers" => ValueSet::PosIntegers,
            "nonnegintegers" => ValueSet::NonNegIntegers,
            "booleans" => ValueSet::Booleans,
            "complexes" => ValueSet::Complexes,
            "rngstates" => ValueSet::RngStates,
            "anything" => ValueSet::Anything,
            _ => ValueSet::Unknown,
        },
        Node::Call(c) => set_call_valueset(inf, &c),
        _ => ValueSet::Unknown,
    }
}

/// A literal numeric bound (used by the `interval` reader above for `neg`).
fn bound_of(inf: &Inferencer<'_, '_>, node: Option<NodeId>) -> Option<f64> {
    match node.map(|n| inf.module.node(n)) {
        Some(Node::Lit(Scalar::Real(r))) => Some(*r),
        Some(Node::Lit(Scalar::Int(i))) => Some(*i as f64),
        Some(Node::Const(sym)) if inf.module.resolve(*sym) == "inf" => Some(f64::INFINITY),
        _ => None,
    }
}

/// The §08 Domain/Support column as a producer table: the support of a
/// distribution constructor, or `Unknown` for a non-distribution.
///
/// Dispatches via the catalogue for all 30 base distributions.  Distributions
/// with `SupportTag::Structural` (currently only Uniform) retain their live
/// code path so the support is computed from the call argument at inference
/// time rather than from a static tag.
fn distribution_support(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> ValueSet {
    use crate::catalogue::{LowerCtx, lower};

    let Some(sig) = crate::catalogue::builtin().base(name) else {
        return ValueSet::Unknown;
    };
    // Structural support: live code path reads the actual set argument.
    // Currently only Uniform; the static catalogue approximation (Unknown) is
    // not used here — the concrete arg-dependent value is what inference needs.
    if support_is_structural(sig) {
        return set_expr_valueset(inf, args.first().map(|a| a.0));
    }
    // All other distributions: lower via the catalogue to get the support ValueSet.
    let ctx = LowerCtx {
        param_dim: &|kwarg| param_dim(inf, args, named, kwarg),
        arg_scalar: &|_| None,
        arg_dim: &|_| Dim::Dynamic,
        arg_type: &|i| arg_ty(args, i).cloned(),
        intern: &crate::catalogue::no_intern,
    };
    let (_ty, vs) = lower(sig, &ctx);
    vs
}

// =====================================================================
// Total-mass classes (Level::Normalization) — spec §11
// =====================================================================

/// Does this `superpose(...)` PROVE a total mass of exactly one, so §11's
/// `%normalized` ("total mass of one") applies rather than merely `%finite`?
///
/// The maths is forced: `superpose` is measure addition (§06 `superpose`,
/// $\nu(A) = M_1(A) + M_2(A) + \ldots$) and `weighted(w, M)` scales by `w`
/// (§06 `weighted`, $\mathrm{d}\nu = w \cdot \mathrm{d}M$), so a superposition
/// of `weighted(w_i, m_i)` with every `m_i` normalized has total mass `Σ w_i`.
/// Prove `Σ w_i = 1` and the sum is a probability measure. §06 words `superpose`
/// as "generally not normalized" and recommends
/// `normalize(superpose(weighted(w1, M1), weighted(w2, M2)))` for a mixture —
/// "generally" is what this reads: the normalize spelling stays correct and
/// stays necessary for everything not proven here.
///
/// This reads the argument SYNTAX of the call, not the argument masses, and it
/// has to: the mass lattice has no "scaled by an unproven factor" element, so
/// `weighted(psi, m)` with a stochastic `psi` folds to `%unknown` and the
/// weights are unrecoverable from the folded class.
///
/// Requires ALL of:
///
/// 1. Every argument is `weighted(w_i, m_i)` (positional spelling) with `m_i`
///    proven `%normalized`.
/// 2. The weights provably sum to one, by exactly two decidable readings —
///    [`literal_weights_sum_to_one`] and [`complement_pair`]. No arithmetic
///    prover, so `superpose(weighted(w, m), weighted(1 - w, m2))` written with
///    two separately-bound halves of one sum is NOT proven.
/// 3. Every weight is provably in [0, 1], so each component is a measure and
///    the sum is a mixture rather than a signed combination.
fn superpose_is_provably_normalized(inf: &Inferencer<'_, '_>, args: &[ArgInfo]) -> bool {
    if args.len() < 2 {
        return false;
    }
    let mut weights = Vec::with_capacity(args.len());
    for (node, _, _) in args {
        let Some((weight, base)) = weighted_parts(inf, *node) else {
            return false;
        };
        if !matches!(
            inf.lookup_type(base),
            Some(Type::Measure {
                mass: Mass::Normalized,
                ..
            })
        ) {
            return false;
        }
        weights.push(weight);
    }
    literal_weights_sum_to_one(inf, &weights) || complement_pair(inf, &weights)
}

/// `(weight, base)` of a `weighted(w, M)` call, looking through binding
/// references so a named component (`signal = weighted(w, m)`) reads the same as
/// an inline one.
///
/// Positional two-argument spelling only. §04 lets a built-in take its arguments
/// by keyword too, so `weighted(weight = w, base = m)` is valid FlatPPL that this
/// declines to prove — conservative, and the exit condition is adding the keyword
/// spelling here, not relaxing anything else.
fn weighted_parts(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<(NodeId, NodeId)> {
    let Node::Call(c) = inf.module.node(resolve_binding_refs(inf, node)) else {
        return None;
    };
    let CallHead::Builtin(op) = c.head else {
        return None;
    };
    // `logweighted` is deliberately absent: its weight is exp(logweight), so
    // proving a sum of one is a different (and non-linear) question.
    if inf.module.resolve(op) != "weighted" || c.args.len() != 2 || !c.named.is_empty() {
        return None;
    }
    Some((c.args[0], c.args[1]))
}

/// Follow `(%ref self x)` hops to the bound right-hand side. Bounded, so a
/// reference cycle (which inference reports separately) cannot spin here.
fn resolve_binding_refs(inf: &Inferencer<'_, '_>, node: NodeId) -> NodeId {
    let mut node = node;
    for _ in 0..16 {
        let Node::Ref(r) = inf.module.node(node) else {
            break;
        };
        if r.ns != RefNs::SelfMod {
            break;
        }
        let Some(binding) = inf.module.binding_by_name(r.name) else {
            break;
        };
        node = inf.module.binding(binding).rhs;
    }
    node
}

/// All weights are non-negative literals whose DECLARED decimal values sum to
/// exactly one.
///
/// "Declared decimal" is the load-bearing choice, and it is neither f64 addition
/// nor the exact value of the stored doubles. Measured, on the three readings of
/// the same weights:
///
/// - `[0.3, 0.7]` — f64 addition gives exactly 1.0, the stored doubles sum to
///   0.99999999999999994…, the declared decimals sum to 1. Accepted here.
/// - `[0.3333333333333333; 3]` — f64 addition gives exactly 1.0 (two roundings
///   land on it), the declared decimals sum to 0.9999999999999999. REJECTED
///   here; an f64 fold would have accepted a mixture whose weights, as written,
///   do not sum to one.
/// - `[0.1; 10]` — the declared decimals sum to 1, so accepted, and this must
///   not depend on f64 addition happening to agree.
///
/// So the proof is exact rational arithmetic over what the model says, never
/// over what a particular float width does with it — FlatPPL mandates no
/// precision, and this verdict must not move with the engine's.
fn literal_weights_sum_to_one(inf: &Inferencer<'_, '_>, weights: &[NodeId]) -> bool {
    fn exact_sum_is_one(inf: &Inferencer<'_, '_>, weights: &[NodeId]) -> Option<bool> {
        let parsed: Option<Vec<(i128, u32)>> = weights
            .iter()
            .map(|&w| decimal_literal(inf.module.node(w)))
            .collect();
        let parsed = parsed?;
        // A negative weight makes the component a signed measure, not a measure.
        if parsed.iter().any(|&(mantissa, _)| mantissa < 0) {
            return Some(false);
        }
        let scale = parsed.iter().map(|&(_, s)| s).max().unwrap_or(0);
        let unit = 10i128.checked_pow(scale)?;
        let mut total: i128 = 0;
        for (mantissa, s) in parsed {
            let lift = 10i128.checked_pow(scale - s)?;
            total = total.checked_add(mantissa.checked_mul(lift)?)?;
        }
        Some(total == unit)
    }
    exact_sum_is_one(inf, weights).unwrap_or(false)
}

/// A built-in whose value is a property of the NODE rather than of the expression:
/// reading the spelling tells you nothing about which value you get, so two
/// occurrences of one spelling may denote two different values.
///
/// §04 is explicit for the parameterized case — `functionof` "traces the ancestor
/// subgraph of its argument back to all leaves of parametric phase — that is, all
/// `elementof` leaves. These **leaf nodes** become the inputs of the reified
/// callable" — so each `elementof` occurrence is its own parameter, exactly as each
/// `draw` is its own stochastic coordinate ("each draw from `m` is a fresh
/// coordinate"). `rand`, `rnginit` and `rngstate` thread explicit randomness, and
/// `external` / `load_data` are compile-time-unknown.
///
/// The set is deliberately wider than freshness alone: two `load_data` calls on the
/// same path do agree, but this predicate exists so a caller can decline to reason
/// about the value at all, and a caller that needs the distinction should not be
/// tempted to shorten the list. Consulted by [`crate::consteval`] (these are
/// `%dynamic`, never a const-eval gap) and by [`is_complement_of`] (structural
/// equality is not value identity across one of these).
pub(crate) fn is_opaque_value_source(name: &str) -> bool {
    matches!(
        name,
        "draw" | "rand" | "elementof" | "external" | "load_data" | "rnginit" | "rngstate"
    )
}

/// Does this expression subtree contain an [`is_opaque_value_source`] call?
///
/// Does NOT follow binding references, and that is the point: one binding is one
/// coordinate, so two `(%ref self psi)` nodes denote the same value however `psi`
/// was produced. Only sources written INSIDE the compared subtree defeat identity.
fn contains_opaque_value_source(inf: &Inferencer<'_, '_>, node: NodeId) -> bool {
    if let Node::Call(c) = inf.module.node(node) {
        if let CallHead::Builtin(op) = c.head {
            if is_opaque_value_source(inf.module.resolve(op)) {
                return true;
            }
        }
    }
    let mut found = false;
    inf.module.for_each_child(node, |child| {
        found = found || contains_opaque_value_source(inf, child);
    });
    found
}

/// A numeric literal as its declared decimal value: `(mantissa, scale)` denotes
/// `mantissa / 10^scale`. `None` for a non-literal, and for a magnitude whose
/// digits would overflow the exact sum.
fn decimal_literal(node: &Node) -> Option<(i128, u32)> {
    match node {
        Node::Lit(Scalar::Int(n)) => Some((*n as i128, 0)),
        Node::Lit(Scalar::Real(r)) if r.is_finite() => {
            // `{}` on an f64 is the shortest decimal that round-trips, i.e. what
            // the model wrote, and never exponent notation.
            let text = format!("{r}");
            let (int_digits, frac_digits) = text.split_once('.').unwrap_or((text.as_str(), ""));
            if int_digits.len() + frac_digits.len() > 18 {
                return None;
            }
            let mantissa: i128 = format!("{int_digits}{frac_digits}").parse().ok()?;
            Some((mantissa, frac_digits.len() as u32))
        }
        _ => None,
    }
}

/// The complement pattern: exactly two weights, `e` and `1 - e` with the same
/// `e`, where `e` is provably in [0, 1]. Then the weights sum to one whatever
/// `e` turns out to be — the one sum-to-one proof that survives a stochastic or
/// parameterized weight.
fn complement_pair(inf: &Inferencer<'_, '_>, weights: &[NodeId]) -> bool {
    let [a, b] = weights else {
        return false;
    };
    is_complement_of(inf, *a, *b) || is_complement_of(inf, *b, *a)
}

/// Is `whole_minus` the expression `1 - part`, for the SAME `part`, with `part`
/// proven to lie in [0, 1]?
///
/// **Structural equality is not value identity, and treating it as such was
/// unsound.** Two inline `draw(Uniform(interval(0.0, 1.0)))` subtrees are
/// syntactically identical and are two independent coordinates (#73: "each draw
/// from `m` is a fresh coordinate"), so `w1 + (1 - w2) = 1` holds only on a
/// probability-zero event — yet the pair typed `%normalized` and lowered as a law
/// with no normalizer. A silently wrong number, and worse than refusing.
///
/// So identity needs one of two things, and structural equality alone is neither:
///
/// - the SAME node, which is identity by construction; or
/// - a subtree with no [`is_opaque_value_source`] inside it, where the spelling
///   does determine the value.
///
/// The legitimate spelling survives because a binding is one coordinate: in
/// `psi ~ Beta(…)` with weights `psi` and `1 - psi`, both compared subtrees are
/// `(%ref self psi)` — no source is written inside either, and the `draw` sits in
/// `psi`'s own binding, which this deliberately does not enter. Two DIFFERENT
/// bindings holding equal draws (`psi` and `phi`) were already unproven, by
/// `Ref` symbol inequality; the inline duplicate was the gap between that control
/// and this test.
fn is_complement_of(inf: &Inferencer<'_, '_>, part: NodeId, whole_minus: NodeId) -> bool {
    let Node::Call(c) = inf.module.node(resolve_binding_refs(inf, whole_minus)) else {
        return false;
    };
    let CallHead::Builtin(op) = c.head else {
        return false;
    };
    if inf.module.resolve(op) != "sub" || c.args.len() != 2 || !c.named.is_empty() {
        return false;
    }
    let subtrahend_is_one = match inf.module.node(c.args[0]) {
        Node::Lit(Scalar::Int(n)) => *n == 1,
        Node::Lit(Scalar::Real(r)) => *r == 1.0,
        _ => false,
    };
    let other = c.args[1];
    let same_value = part == other
        || (!contains_opaque_value_source(inf, part) && !contains_opaque_value_source(inf, other));
    subtrahend_is_one
        && inf.module.structural_eq(part, other)
        && same_value
        // [0, 1] is exactly `unitinterval` (§11 value sets); `subset_of` proves
        // containment or answers false, so an unconstrained weight is unproven.
        && inf
            .lookup_valueset(part)
            .subset_of(&ValueSet::UnitInterval)
}

/// Fill the `%mass` slot of a measure/kernel-typed call result, per the §06
/// composition rules. `normalize` on a measure with statically known zero or
/// infinite mass is a static error (spec: the result is undefined).
pub(crate) fn fill_mass(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    call: &Call,
    callee: Option<&(NodeId, Type)>,
    ty: Type,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    // Only measure types carry a deferred mass to fill; kernels and user
    // calls were filled at construction (their mass rides the callee).
    let Type::Measure { domain, mass } = ty else {
        return ty;
    };
    if mass != Mass::Deferred {
        return Type::Measure { domain, mass };
    }
    if callee.is_some() || call.inputs.is_some() {
        // User calls (including applied §09 catalogue distribution references)
        // had their mass set at construction — a catalogue distribution carries
        // its `MassTag`-derived mass (Normalized/Finite) out of
        // `catalogue_call_type`, so it is already concrete and was returned by
        // the `mass != Deferred` guard above. Reaching here means a deferred
        // mass that the call site cannot refine; pass it through unchanged.
        return Type::Measure { domain, mass };
    }
    let CallHead::Builtin(op) = call.head else {
        return Type::Measure { domain, mass };
    };
    let name = inf.module.resolve(op).to_string();

    let arg_mass = |i: usize| match arg_ty(args, i) {
        Some(Type::Measure { mass, .. }) => *mass,
        // Kernels carry a mass class too (a `Normalized` kernel is a Markov /
        // probability kernel) — chain/trajectory ops read it.
        Some(Type::Kernel { mass, .. }) => *mass,
        _ => Mass::Unknown,
    };

    let mass = match name.as_str() {
        // `"lawof"` passes `mass` through unchanged: `lawof_type` already set
        // the result mass (the gate-admitted argument's own mass, per the
        // no-laundering rider), so this must NOT fall to the `_` catchall
        // below — that catchall assumes "every unlisted head is a §08
        // distribution, hence `%normalized`", which was harmlessly true only
        // while `lawof_type` always produced `%normalized` itself (the guard
        // above short-circuits unless `ty`'s mass is `%deferred`). Now that
        // `lawof_type` can legitimately produce `%deferred`, an absent arm
        // would re-launder it to `%normalized` right where the no-laundering
        // rider forbids it.
        "lawof" => mass,
        // Every §08 distribution is a probability measure.
        // `Dirac(value)` is a point-mass probability measure (total mass 1).
        "Dirac" => Mass::Normalized,
        // Reference measures: finite on a bounded support, infinite (but
        // boundedly finite) on an unbounded one.
        "Lebesgue" | "Counting" => {
            let support_node = lebesgue_counting_support_node(inf, args, named);
            match set_expr_valueset(inf, support_node).is_bounded() {
                Some(true) => Mass::Finite,
                Some(false) => Mass::LocallyFinite,
                None => Mass::Unknown,
            }
        }
        "iid" => match arg_mass(0) {
            Mass::Normalized => Mass::Normalized,
            Mass::Null => Mass::Null,
            Mass::Finite => Mass::Finite,
            Mass::LocallyFinite => Mass::LocallyFinite,
            _ => Mass::Unknown,
        },
        "joint" => joint_mass(inf, args, named),
        // `restrict` shares truncate's support-restriction mass behaviour: the
        // result is a sub-measure, so a probability/finite measure becomes
        // merely finite, and an infinite measure stays finite only on a bounded
        // restriction set.
        "truncate" | "restrict" => match arg_mass(0) {
            Mass::Null => Mass::Null,
            Mass::Normalized | Mass::Finite => Mass::Finite,
            Mass::LocallyFinite => {
                match set_expr_valueset(inf, args.get(1).map(|a| a.0)).is_bounded() {
                    Some(true) => Mass::Finite,
                    _ => Mass::Unknown,
                }
            }
            _ => Mass::Unknown,
        },
        // Pushforward through a (measurable) map preserves total mass (spec §06
        // image measure): `pushfwd(f, M)` keeps M's mass, `locscale(M, …)` — an
        // affine pushforward — keeps M's mass.
        "pushfwd" => arg_mass(1),
        "locscale" => arg_mass(0),
        // `markovchain(kernel, …)` / `kscan(kernel, …)`: a trajectory of a
        // normalized (Markov) kernel is itself a probability measure; a
        // non-normalized step kernel gives an unknown total mass.
        "markovchain" | "kscan" => match arg_mass(0) {
            Mass::Normalized => Mass::Normalized,
            Mass::Null => Mass::Null,
            _ => Mass::Unknown,
        },
        // `kchain`: a Kleisli chain of probability components (base measure +
        // Markov kernels) is a probability measure; otherwise the total mass is
        // not statically known (the bind carries a generally-intractable
        // marginalization integral). (`jointchain` has its own mass arm below.)
        "kchain" => {
            if (0..args.len()).all(|i| matches!(arg_mass(i), Mass::Normalized)) {
                Mass::Normalized
            } else {
                Mass::Unknown
            }
        }
        // `superpose(M1, M2, …)` is measure addition Z = Σ Zi (spec §06): the
        // sum of finite masses is finite but generally not normalized; any
        // infinite component makes the sum infinite; an Unknown taints the sum.
        //
        // "Generally" has one decidable exception, checked first because it
        // recovers a class the folded masses have already lost: a mixture whose
        // component weights PROVABLY sum to one is normalized, even when each
        // `weighted` component folded to `%unknown`.
        "superpose" if superpose_is_provably_normalized(inf, args) => Mass::Normalized,
        "superpose" => {
            let masses: Vec<Mass> = (0..args.len()).map(arg_mass).collect();
            if masses.iter().any(|m| matches!(m, Mass::Unknown)) {
                Mass::Unknown
            } else if masses.iter().any(|m| matches!(m, Mass::LocallyFinite)) {
                Mass::LocallyFinite
            } else if masses.iter().all(|m| matches!(m, Mass::Null)) {
                Mass::Null
            } else {
                Mass::Finite
            }
        }
        // A fixed scalar weight rescales: classes survive, except that an
        // unknown constant demotes `%normalized` to `%finite`.
        "weighted" | "logweighted" => {
            let base = arg_mass(1);
            if base == Mass::Null {
                Mass::Null
            } else if matches!(
                (arg_ty(args, 0), args.first().map(|(_, _, p)| *p)),
                (Some(Type::Scalar(_)), Some(Phase::Fixed))
            ) {
                match base {
                    Mass::Normalized | Mass::Finite => Mass::Finite,
                    Mass::LocallyFinite => Mass::LocallyFinite,
                    _ => Mass::Unknown,
                }
            } else {
                Mass::Unknown
            }
        }
        "normalize" => match arg_mass(0) {
            Mass::Null => {
                inf.diags.push(crate::Diagnostic::error_at(
                    id,
                    "`normalize` of a measure with zero total mass is undefined (spec §06)",
                ));
                return Type::Failed("normalize of a zero-mass measure".into());
            }
            Mass::LocallyFinite => {
                inf.diags.push(crate::Diagnostic::error_at(
                    id,
                    "`normalize` of a measure with infinite total mass is undefined (spec §06)",
                ));
                return Type::Failed("normalize of an infinite-mass measure".into());
            }
            _ => Mass::Normalized,
        },
        "bayesupdate" => Mass::Unknown,
        "jointchain" => {
            // `jointchain(M, K1, …, Kn)` spec §06: the result carries the base
            // measure's mass class (component 0) provided every kernel (components
            // 1..n) is Normalized.  A Finite base + Normalized kernels ⇒ Finite
            // result; a Normalized base + Normalized kernels ⇒ Normalized result.
            // If any kernel is not Normalized the total mass is generally
            // intractable ⇒ Unknown.
            let named_mass = |t: &Type| match t {
                Type::Measure { mass, .. } => *mass,
                Type::Kernel { mass, .. } => *mass,
                _ => Mass::Unknown,
            };
            let (base_mass, kernels_normalized): (Mass, bool) = if !named.is_empty() {
                let base = named
                    .first()
                    .map(|(_, _, t, _)| named_mass(t))
                    .unwrap_or(Mass::Unknown);
                let all_kernels_norm = named
                    .iter()
                    .skip(1)
                    .all(|(_, _, t, _)| matches!(named_mass(t), Mass::Normalized));
                (base, all_kernels_norm)
            } else {
                let n = args.len();
                if n == 0 {
                    (Mass::Unknown, true)
                } else {
                    let base = arg_mass(0);
                    let all_kernels_norm = (1..n).all(|i| matches!(arg_mass(i), Mass::Normalized));
                    (base, all_kernels_norm)
                }
            };
            if kernels_normalized {
                base_mass
            } else {
                Mass::Unknown
            }
        }
        // A §08 distribution constructor (this arm is reached only for
        // measure-typed results, i.e. recognized distributions).
        _ => Mass::Normalized,
    };
    Type::Measure { domain, mass }
}

/// The total-mass class of a `joint`, measure or kernel: [`product_mass`] over
/// the components, qualified by per-component trace cleanliness.
///
/// One fold for both, because §06 gives the kernel case no rule of its own — at
/// each input the fan-out kernel's output IS a measure-`joint`, and §11 makes a
/// kernel's `%mass` "the total-mass class of the output measure, uniform over all
/// inputs" (`kernel-joint-q4-maths.md` §7, Q5). A kernel component is read for its
/// own class the same way [`fill_mass`]'s `arg_mass` and `jointchain`'s arm read
/// one: a `%normalized` kernel is a Markov kernel, so a fan-out of Markov kernels
/// is a Markov kernel.
///
/// Keyword and positional `joint` are mutually exclusive forms ([`joint_type`]):
/// the split is mirrored here so both spellings of the same joint fold
/// identically.
fn joint_mass(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo], named: &[NamedInfo]) -> Mass {
    let (masses, not_clean): (Vec<Mass>, Vec<bool>) = if !named.is_empty() {
        named
            .iter()
            .map(|(_, node, t, _)| {
                (
                    component_mass(inf, *node, t),
                    !joint_component_is_trace_clean(inf, *node, 0),
                )
            })
            .unzip()
    } else {
        args.iter()
            .map(|(node, t, _)| {
                (
                    component_mass(inf, *node, t),
                    !joint_component_is_trace_clean(inf, *node, 0),
                )
            })
            .unzip()
    };
    product_mass(&masses, &not_clean)
}

/// The mass class [`joint_mass`] folds for one component.
///
/// A measure component is read straight off its type, as it always was. A KERNEL
/// component is read only when its own type rule SET that mass; otherwise the
/// fold refuses to trust it and contributes `%unknown`.
///
/// The distrust is not conservatism, it is a live defect on the other side.
/// `fill_mass` returns early unless the type is `Type::Measure`, so an op whose
/// type rule lifts pointwise to a kernel never reaches its own mass arm and keeps
/// the BASE's class: `truncate(kernelof(a1, z = z), interval(0.0, 5.0))` reads
/// `%normalized` where the measure version correctly reads `%finite`. Folding that
/// unchanged published a wrong STRONGER class — a `joint` of two such components
/// read `%normalized` instead of Q5's `%unknown`, and `kchain` carried the
/// `%normalized` onto a MEASURE, past the `normalize`/`draw`/`rand` gates an
/// unnormalized measure must not pass.
///
/// Fixing the lift itself (in `fill_mass`) is the better end state and is carded
/// separately: it touches every measure-to-measure arm and flips kernel classes
/// across the corpus, which carries its own repin risk. This read is chosen
/// instead because it cannot move any class that exists today — a `Type::Kernel`
/// component only reaches a `joint` through [`kernel_joint_type`], which is new,
/// so measure-joint output is untouched — and it fails toward the weaker class.
///
/// The trusted heads are exactly the rules that write a kernel's mass themselves:
/// `kernelof`/`functionof` (`reification_type`, from the body), `lawof`
/// (`lawof_type`, propagating the argument's own gate-admitted mass), `joint` itself
/// (`kernel_joint_type`, recursively through this same fold), and `ksuperpose`
/// ([`ksuperpose_type`], from §06's $\sum_i w_i$ mass sentence). Everything else —
/// including a `disintegrate` tuple element and a cross-module kernel, whose
/// masses ARE set correctly but not by a head this can see — reads `%unknown`.
fn component_mass(inf: &Inferencer<'_, '_>, node: NodeId, t: &Type) -> Mass {
    match t {
        Type::Measure { mass, .. } => *mass,
        Type::Kernel { mass, .. } if kernel_mass_is_own_rules(inf, node) => *mass,
        _ => Mass::Unknown,
    }
}

/// Whether a kernel-typed component's `%mass` was set by its own type rule, and
/// so may be folded — see [`component_mass`] for why the question has to be asked
/// at all. Follows self-module refs to the head that produced the value.
fn kernel_mass_is_own_rules(inf: &Inferencer<'_, '_>, node: NodeId) -> bool {
    let mut node = node;
    let mut depth = 0u32;
    loop {
        if depth > 64 {
            return false; // safe: "not provably its own rule's mass"
        }
        depth += 1;
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let Some(binding) = inf.module.binding_by_name(r.name) else {
                    return false;
                };
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(c) => {
                let CallHead::Builtin(op) = c.head else {
                    return false;
                };
                return matches!(
                    inf.module.resolve(op),
                    "kernelof" | "functionof" | "lawof" | "joint" | "ksuperpose"
                );
            }
            _ => return false,
        }
    }
}

/// The mass of a `joint`'s components (spec §06 `joint` entry: "a stochastic
/// node shared between component traces … remains a single node of the
/// composed trace. Components that share no stochastic node are independent,
/// and their `joint` is the product measure"). For trace-disjoint components
/// the product-of-classes rule below is exact. For components sharing a
/// stochastic ancestor it is exact only up to one non-normalized member:
/// `%finite`x`%finite` composed through a shared ancestor can be infinite
/// (Student-t/`y^2` counterexample, kernel-joint-q4-maths.md §7), so once two
/// or more components that MAY share an ancestor are non-normalized, no class
/// stronger than `%unknown` is statically justified.
///
/// `not_clean[i]` says component `i` is not PROVABLY free of any stochastic
/// trace node (see [`joint_component_is_trace_clean`]) — i.e. it may
/// participate in sharing. A component that IS provably clean cannot share
/// with anything (spec §04 "Identity law": `joint(m, m)` over a bare
/// constructor `m` is the product of two independent draws, never the
/// diagonal) and so is exempt from the degrade below regardless of its own
/// mass class — only the "may share" components are counted toward the
/// two-or-more threshold.
///
/// An EMPTY component list (`joint()`) is `%deferred`, not `%normalized`:
/// `joint_type`'s domain arm already leaves a zero-component `joint`'s domain
/// `%deferred` (nothing resolves the variate shape), so the mass side
/// matching that honestly, rather than vacuously satisfying `all()` and
/// claiming a definite class the domain itself does not support, keeps the
/// two slots consistent. `joint()`'s legality is a separate, unaddressed
/// question (spec §06 spells the construct `joint(M1, M2, ...)`).
fn product_mass(masses: &[Mass], not_clean: &[bool]) -> Mass {
    use Mass::*;
    // `zip` below truncates silently on a length mismatch, which would
    // under-count the degrade and hand back a class stronger than justified.
    // The single call site builds both slices from one `unzip`, so they
    // always agree; this documents the invariant for the next caller.
    debug_assert_eq!(masses.len(), not_clean.len());
    if masses.is_empty() {
        return Deferred;
    }
    if masses.contains(&Null) {
        return Null;
    }
    if masses.iter().all(|m| *m == Normalized) {
        return Normalized;
    }
    let ambiguous_nonnormalized = masses
        .iter()
        .zip(not_clean)
        .filter(|(m, dirty)| !matches!(m, Normalized | Deferred) && **dirty)
        .count();
    if ambiguous_nonnormalized >= 2 {
        return Unknown;
    }
    if masses.iter().all(|m| matches!(m, Normalized | Finite)) {
        return Finite;
    }
    if masses
        .iter()
        .all(|m| matches!(m, Normalized | Finite | LocallyFinite))
    {
        return LocallyFinite;
    }
    Unknown
}

/// Whether a `joint` COMPONENT provably carries no stochastic trace node,
/// following self-module refs the way [`law_phase`] does (bounded by the same
/// `depth` cap). Spec §04 "Trace of the reified law": a stochastic node
/// enters a `joint`'s composed trace only through a REIFIED component
/// (`lawof`/`kernelof`) or, per §06's `joint` entry, "a stochastic
/// constructor parameter". A subtree containing neither channel cannot carry
/// or close over a stochastic node, so it cannot share one with any sibling
/// component (§04 "Identity law": `joint(m, m)` over a bare constructor `m`
/// is the product of two independent draws, never the diagonal).
///
/// The disqualifier list is `draw`/`rand` (§04 "Phases": "`draw` nodes are
/// stochastic" — the only source of a stochastic node at all) plus the three
/// ways a value can carry or close over a REIFIED trace — `lawof`,
/// `kernelof`, `functionof` — plus `rnginit`/`rngstate` (an RNG state is not
/// itself a stochastic node, but `rand` consuming a shared state is close
/// enough to correlation that refusing to call it clean costs nothing).
/// Deliberately NOT [`is_opaque_value_source`]'s whole catalogue: that
/// predicate answers a different question (value identity), and its
/// `elementof`/`external`/`load_data` members are §04-classified
/// *parameterized* or *fixed*, never stochastic — kernel-joint-q4-maths.md §8
/// is explicit that "a shared input name is a shared value, not a shared
/// stochastic node". Any of the disqualifiers ANYWHERE in the subtree answers
/// `false` ("not provably clean"), never "definitely shares" — this only ever
/// RULES OUT sharing, never proves it, so it is not the ancestry oracle the
/// wave brief forbids inventing: it never compares two components against
/// each other, only asks a local question of one.
///
/// Memoized (`memo`) because the walk follows refs through the binding DAG,
/// which is a DAG and not a tree: an unmemoized walk revisits a
/// diamond-shared binding once per PATH to it, which is exponential in depth
/// on a chain of repeated sub-expressions. Caching each node's answer makes
/// the walk linear in the number of distinct nodes visited. The `depth > 64`
/// bailout is intentionally NOT cached — it is a conservative fallback for a
/// pathologically deep single path, not that node's true answer, and caching
/// it could make a later, shorter path to the same node see a stale `false`.
fn joint_component_is_trace_clean(inf: &Inferencer<'_, '_>, node: NodeId, depth: u32) -> bool {
    fn walk(
        inf: &Inferencer<'_, '_>,
        node: NodeId,
        depth: u32,
        memo: &mut std::collections::HashMap<NodeId, bool>,
    ) -> bool {
        if let Some(&clean) = memo.get(&node) {
            return clean;
        }
        if depth > 64 {
            return false; // safe: "not provably clean" on a cycle/depth cap
        }
        if let Node::Ref(r) = inf.module.node(node) {
            let clean = if r.ns != flatppl_core::RefNs::SelfMod {
                false // cross-module: conservatively not provably clean
            } else {
                match inf.module.binding_by_name(r.name) {
                    Some(b) => {
                        let rhs = inf.module.binding(b).rhs;
                        walk(inf, rhs, depth + 1, memo)
                    }
                    None => false,
                }
            };
            memo.insert(node, clean);
            return clean;
        }
        if let Node::Call(c) = inf.module.node(node) {
            match c.head {
                CallHead::Builtin(op) => {
                    let name = inf.module.resolve(op);
                    if matches!(
                        name,
                        "draw"
                            | "rand"
                            | "rnginit"
                            | "rngstate"
                            | "lawof"
                            | "kernelof"
                            | "functionof"
                    ) {
                        memo.insert(node, false);
                        return false;
                    }
                }
                CallHead::User(_) => {
                    memo.insert(node, false); // user-callable application: conservative
                    return false;
                }
            }
        }
        let mut clean = true;
        inf.module.for_each_child(node, |child| {
            clean = clean && walk(inf, child, depth + 1, memo);
        });
        memo.insert(node, clean);
        clean
    }
    walk(inf, node, depth, &mut std::collections::HashMap::new())
}

/// Broadcasting a kernel over data cells: an independent product per cell.
fn broadcast_mass(cell: Mass) -> Mass {
    match cell {
        Mass::Normalized => Mass::Normalized,
        Mass::Null => Mass::Null,
        Mass::Finite => Mass::Finite,
        _ => Mass::Unknown,
    }
}

// =====================================================================
// Test helpers (not part of the normal public API)
// =====================================================================

/// For test use: the variate domain of a named distribution, with
/// `param_dim` provided by the caller rather than inferred from a live Module.
/// Returns `None` for non-distributions, matching the production function.
#[cfg(test)]
pub(crate) fn distribution_domain_static(
    name: &str,
    param_dim: &dyn Fn(&str) -> Dim,
) -> Option<Type> {
    use ScalarType::*;
    let scalar = |s: ScalarType| Some(Type::Scalar(s));
    let dynmat = || {
        Some(Type::Array {
            shape: Box::new([Dim::Dynamic, Dim::Dynamic]),
            elem: Box::new(Type::Scalar(Real)),
        })
    };
    match name {
        "Normal" | "GeneralizedNormal" | "Cauchy" | "StudentT" | "Logistic" | "LogNormal"
        | "Exponential" | "Gamma" | "Weibull" | "Pareto" | "InverseGamma" | "Beta"
        | "ChiSquared" | "VonMises" | "Laplace" | "Uniform" => scalar(Real),
        // Bernoulli: spec §08 "Domain/Support: integers/booleans".
        // Legacy ops.rs returned Boolean — that is a legacy bug; oracle now
        // reflects the spec-correct value (Integer) to match the catalogue.
        "Bernoulli" => scalar(Integer),
        "Categorical" | "Categorical0" | "Binomial" | "Geometric" | "NegativeBinomial"
        | "NegativeBinomial2" | "Poisson" => scalar(Integer),
        "MvNormal" => Some(Type::Array {
            shape: Box::new([param_dim("mu")]),
            elem: Box::new(Type::Scalar(Real)),
        }),
        "Dirichlet" => Some(Type::Array {
            shape: Box::new([param_dim("alpha")]),
            elem: Box::new(Type::Scalar(Real)),
        }),
        "Multinomial" => Some(Type::Array {
            shape: Box::new([param_dim("p")]),
            elem: Box::new(Type::Scalar(Integer)),
        }),
        "Wishart" | "InverseWishart" | "LKJ" | "LKJCholesky" => dynmat(),
        _ => None,
    }
}

/// For test use: the support `ValueSet` of a named distribution, with
/// `param_dim` provided by the caller. Returns `ValueSet::Unknown` for
/// non-distributions or arg-dependent supports (Uniform, Wishart family).
#[cfg(test)]
pub(crate) fn distribution_support_static(name: &str, param_dim: &dyn Fn(&str) -> Dim) -> ValueSet {
    use ValueSet::*;
    match name {
        // Uniform: support is structural (the set arg passed at the call site,
        // evaluated by set_expr_valueset at inference time). This oracle returns
        // Unknown — the correct static approximation — so the faithfulness test
        // can verify it matches the catalogue's Structural tag (which also lowers
        // to Unknown). The real arg-dependent behavior is guarded by the
        // dedicated `uniform_support_is_the_argument_set` test.
        "Uniform" => Unknown,
        "Normal" | "GeneralizedNormal" | "Cauchy" | "StudentT" | "Logistic" | "VonMises"
        | "Laplace" => Reals,
        "LogNormal" | "InverseGamma" | "Pareto" => PosReals,
        "Exponential" | "Weibull" | "Gamma" | "ChiSquared" => NonNegReals,
        "Beta" => UnitInterval,
        "Bernoulli" => Booleans,
        "Categorical" => PosIntegers,
        "Categorical0" | "Binomial" | "Geometric" | "NegativeBinomial" | "NegativeBinomial2"
        | "Poisson" => NonNegIntegers,
        "MvNormal" => CartPow(Box::new(Reals), param_dim("mu")),
        "Dirichlet" => StdSimplex(param_dim("alpha")),
        "Multinomial" => CartPow(Box::new(NonNegIntegers), param_dim("p")),
        // not in distribution_support — legacy returns Unknown
        "Wishart" | "InverseWishart" | "LKJ" | "LKJCholesky" => Unknown,
        _ => Unknown,
    }
}

/// For test use: the expected result type of a per-name function, mirroring
/// what the old per-name call_rule arms produced.  This is the static oracle
/// for the catalogue faithfulness test: it must match `function_result` for
/// every name in the migrated set.
///
/// `arg_scalar` simulates the caller's arg type at position 0.
/// Returns `None` for names that were never in the per-name arm set.
#[cfg(test)]
pub(crate) fn function_type_static(name: &str, arg0_scalar: Option<ScalarType>) -> Option<Type> {
    use ScalarType::*;
    let real_or_cplx = |s: Option<ScalarType>| match s {
        Some(Complex) => Type::Scalar(Complex),
        _ => Type::Scalar(Real),
    };
    match name {
        // scalar-integer output
        "floor" | "ceil" | "round" | "integer" => Some(Type::Scalar(Integer)),
        "div" | "mod" => Some(Type::Scalar(Integer)),
        "lengthof" | "length" => Some(Type::Scalar(Integer)),
        // scalar-real output
        // (divide and mean are NOT here: they are structural — divide promotes
        // its two operands, mean reduces to the array element type — handled
        // in call_rule, not the catalogue.)
        "logdensityof" | "densityof" => Some(Type::Scalar(Real)),
        "l1norm" | "l2norm" | "logsumexp" => Some(Type::Scalar(Real)),
        // scalar-complex output
        "cis" | "complex" => Some(Type::Scalar(Complex)),
        // scalar-boolean output
        "equal" | "unequal" | "lt" | "le" | "gt" | "ge" | "in" | "land" | "lor" | "lnot"
        | "isfinite" | "isinf" | "isnan" | "iszero" => Some(Type::Scalar(Boolean)),
        // real_or_complex: exp/log/sqrt/trig and friends
        "exp" | "log" | "log2" | "log10" | "sqrt" | "sin" | "cos" | "tan" | "asin" | "acos"
        | "atan" | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" | "log1p" | "expm1"
        | "gamma" | "loggamma" | "logit" | "invlogit" | "probit" | "invprobit" | "conj" => {
            Some(real_or_cplx(arg0_scalar))
        }
        // abs / abs2: |z| and |z|² are always REAL even for complex input
        // (spec §07: |·| maps ℂ → ℝ). Legacy ops.rs used real_or_complex which
        // incorrectly returned Complex for complex input; the catalogue
        // DomainMap(Complex→Real) is spec-correct. The test oracle follows spec.
        "abs" | "abs2" => Some(Type::Scalar(Real)),
        _ => None,
    }
}

/// The strongest common containing set of several element sets: their shared
/// value if equal, else the strongest named scalar set that contains all
/// (widening, strongest first). `None` when nothing fits.
fn join_scalar_sets(sets: &[ValueSet]) -> Option<ValueSet> {
    let first = sets.first()?;
    if sets.iter().all(|s| s == first) && *first != ValueSet::Unknown {
        return Some(first.clone());
    }
    const CANDIDATES: &[ValueSet] = &[
        ValueSet::PosIntegers,
        ValueSet::NonNegIntegers,
        ValueSet::Integers,
        ValueSet::UnitInterval,
        ValueSet::PosReals,
        ValueSet::NonNegReals,
        ValueSet::Reals,
        ValueSet::Booleans,
        ValueSet::Complexes,
    ];
    CANDIDATES
        .iter()
        .find(|c| sets.iter().all(|s| s.subset_of(c)))
        .cloned()
}

#[cfg(test)]
mod cat_compose_tests {
    //! Unit coverage for the §06 cat-composition helper (the `joint_likelihood`
    //! obstype rule). The scalar→vector branch is also exercised end-to-end by
    //! the `joint_likelihood_unions_inputs_and_cats_obstype` golden test; these
    //! cover the array-concat / mixed-class / deferred branches directly.
    use super::*;

    fn arr(n: Dim, elem: ScalarType) -> Type {
        Type::Array {
            shape: Box::new([n]),
            elem: Box::new(Type::Scalar(elem)),
        }
    }

    #[test]
    fn scalars_make_a_length_n_vector() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[Type::Scalar(Real), Type::Scalar(Real)]),
            arr(Dim::Static(2), Real)
        );
    }

    #[test]
    fn arrays_concatenate_their_lengths() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[arr(Dim::Static(2), Real), arr(Dim::Static(3), Real)]),
            arr(Dim::Static(5), Real)
        );
    }

    #[test]
    fn a_dynamic_length_makes_the_concat_dynamic() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[arr(Dim::Dynamic, Real), arr(Dim::Static(3), Real)]),
            arr(Dim::Dynamic, Real)
        );
    }

    #[test]
    fn mixed_shape_classes_defer() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[Type::Scalar(Real), arr(Dim::Static(2), Real)]),
            Type::Deferred
        );
    }

    #[test]
    fn a_deferred_component_propagates() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[Type::Scalar(Real), Type::Deferred]),
            Type::Deferred
        );
    }

    #[test]
    fn an_empty_list_defers() {
        assert_eq!(cat_compose(&[]), Type::Deferred);
    }
}
