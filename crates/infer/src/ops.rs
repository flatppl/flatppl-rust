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
        "record" => match (named.is_empty(), args) {
            (true, [(_, Type::Table { columns, nrows }, _)]) => record_from_table(columns, *nrows),
            _ => Type::Record(named.iter().map(|(n, _, t, _)| (*n, t.clone())).collect()),
        },
        // `table(r)` auto-splats a single record-of-vectors into columns (spec
        // §03); otherwise a table of its named columns.
        "table" => match (named.is_empty(), args) {
            (true, [(node, Type::Record(fields), _)]) => {
                let cols: Vec<(Symbol, &Type, NodeId)> =
                    fields.iter().map(|(n, t)| (*n, t, *node)).collect();
                build_table(inf, &cols)
            }
            _ => table_type(inf, named),
        },
        "rowstack" => rowstack_type(arg_ty(args, 0)),
        "get" => get_type(inf, args, /*base=*/ 1),
        "get0" => get_type(inf, args, /*base=*/ 0),
        "indicesof" | "indicesof0" => Type::Array {
            shape: Box::new([Dim::Dynamic]),
            elem: Box::new(Type::Scalar(ScalarType::Integer)),
        },
        // `sum` / `prod` / `mean` reduce a real/complex array to its element
        // type (spec §07 Reductions): mean of a complex array is complex.
        // (NOT a constant Scalar(Real); legacy ops.rs returned Real always.)
        "sum" | "prod" | "mean" => reduce_type(arg_ty(args, 0)),
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
            let elem = Type::Scalar(
                arg_ty(args, 2)
                    .and_then(elem_scalar_kind_of)
                    .unwrap_or(ScalarType::Real),
            );
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
            let nl = args.get(1).and_then(|a| resolve_fixed_int(inf, a.0, 0));
            let nt = args.get(2).and_then(|a| resolve_fixed_int(inf, a.0, 0));
            match (arg_ty(args, 0), nl, nt) {
                (Some(Type::Array { shape, elem }), Some(nl), Some(nt)) if nl >= 0 && nt >= 0 => {
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
        // cat(x, y, …) concatenates values of the same structural kind: scalars
        // → a rank-1 vector of that kind; arrays → the same rank/element as the
        // first argument (one axis grows, so sizes are dynamic).
        "cat" => match arg_ty(args, 0) {
            Some(Type::Scalar(s)) => Type::Array {
                shape: Box::new([Dim::Dynamic]),
                elem: Box::new(Type::Scalar(*s)),
            },
            Some(t @ Type::Array { .. }) => with_dynamic_dims(t),
            _ => Type::Deferred,
        },

        // ---- parameters / inputs (spec §04) ----
        "elementof" | "external" => set_element_type(inf, args.first().map(|a| a.0)),
        // `load_data(source, valueset)` (spec §07): a vector/table of the
        // declared `valueset`'s element type. The row count is not statically
        // known (spec §11 "a common source of dynamic row counts"), so the
        // leading dim is `%dynamic`. `valueset` is the keyword or the second
        // positional arg (after `source`). A scalar / cartpow / stdsimplex
        // valueset gives a vector; a cartprod (table) valueset is left deferred.
        "load_data" => {
            let vs = named_or_positional_node(inf.module, named, args, "valueset", 1);
            match set_element_type(inf, vs) {
                Type::Deferred => Type::Deferred,
                elem => Type::Array {
                    shape: Box::new([Dim::Dynamic]),
                    elem: Box::new(elem),
                },
            }
        }

        // ---- measure algebra (spec §06) ----
        "lawof" => Type::Measure {
            domain: Box::new(args.first().map_or(Type::Any, |(_, t, _)| t.clone())),
            mass: Mass::Deferred,
        },
        "draw" => measure_domain(arg_ty(args, 0)),
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
        "restrict" | "superpose" | "locscale" => fresh_measure(arg_ty(args, 0)),
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
        // Reference measures (spec §06): measures over their support set.
        "Lebesgue" | "Counting" => Type::Measure {
            domain: Box::new(set_element_type(inf, args.first().map(|a| a.0))),
            mass: Mass::Deferred,
        },
        // `Dirac(value = v)` (spec §06) is the point-mass probability measure at
        // `v`, for any variate type: the domain is `v`'s type. `value` is given
        // as the named kwarg (spec form) or positionally. Mass is normalized
        // (total mass 1) — set in `fill_mass`.
        "Dirac" => {
            let v = arg_ty(args, 0).cloned().or_else(|| {
                named
                    .iter()
                    .find(|(n, _, _, _)| inf.module.resolve(*n) == "value")
                    .map(|(_, _, t, _)| t.clone())
            });
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
        "joint" => {
            // Positional `joint` cats the component variates; mixing shape
            // classes (e.g. a scalar with a vector measure) is a static error
            // (spec §06), not a silently-deferred domain. Only fire when every
            // component is a fully-resolved measure (else defer quietly).
            if named.is_empty() {
                let domains: Option<Vec<Type>> = args
                    .iter()
                    .map(|(_, t, _)| match t {
                        Type::Measure { domain, .. } => Some((**domain).clone()),
                        _ => None,
                    })
                    .collect();
                if let Some(domains) = domains {
                    if cat_is_mixed(&domains) {
                        inf.diags.push(crate::Diagnostic::error_at(
                            id,
                            "joint components must share a shape class (spec §06): mixing \
                             scalars, vectors, and records is a static error",
                        ));
                    }
                }
            }
            joint_type(args, named)
        }
        "likelihoodof" => likelihood_type(inf, args),
        "joint_likelihood" => joint_likelihood_type(args),

        // ---- explicit RNG (spec §07) ----
        "rnginit" => Type::RngState,
        "rand" => match measure_domain(arg_ty(args, 1)) {
            Type::Deferred => Type::Deferred,
            domain => Type::Tuple(Box::new([domain, Type::RngState])),
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
        // bare distribution constructor, from `kernel_variate` (the catalogue). The
        // no-dims case is typed here; the optional `n, m, …` dims that array-ify the
        // variate are a follow-up.
        "builtin_sample" => {
            let k = args.get(1);
            let variate = k
                .and_then(|(n, t, _)| component_variate(inf, *n, t))
                .or_else(|| {
                    k.and_then(|(n, _, _)| kernel_variate(inf, *n, args.get(2).map(|a| a.0)))
                });
            match variate {
                Some(v) => Type::Tuple(Box::new([v, Type::RngState])),
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
        // Per-name functions whose result is a pure scalar (constant, SameScalarKind,
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
/// matrix-multiply forms over FLAT rank-2 matrices — matrix·matrix
/// (`[m,k]·[k,n] → [m,n]`) and matrix·vector (`[m,k]·[k] → [m]`). A statically
/// provable inner-dimension mismatch is a shape error (`%failed`). The remaining
/// §07 forms — transposed-vector·vector (dot) and vector·transposed-vector
/// (outer), and matmul over matrices represented as nested rank-1 arrays — are
/// not yet typed and stay `%deferred` (honest: no guessed shape).
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
fn reduce_type(a: Option<&Type>) -> Type {
    match a {
        Some(Type::Array { elem, .. }) => elem.as_ref().clone(),
        Some(Type::Any) => Type::Any,
        _ => Type::Deferred,
    }
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
                    let elem = set_element_type(inf, set_arg);
                    let shape = size_arg.map_or_else(
                        || Box::new([Dim::Dynamic]) as Box<[Dim]>,
                        |n| count_dims(inf, n),
                    );
                    Type::Array {
                        shape,
                        elem: Box::new(elem),
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
                        // Mixing shape classes (a scalar set with a vector set)
                        // is a static error — §03 cartprod mirrors §06 joint, and
                        // §07 `cat` forbids that concatenation.
                        if cat_is_mixed(&parts) {
                            inf.diags.push(crate::Diagnostic::error_at(
                                node,
                                "cartprod components must share a shape class (spec §03, \
                                 mirroring §06 `joint`): mixing scalars and vectors is a \
                                 static error",
                            ));
                        }
                        cat_compose(&parts)
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
fn iid_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Type {
    let domain = match arg_ty(args, 0) {
        Some(Type::Measure { domain, .. }) => domain.as_ref().clone(),
        _ => return Type::Deferred,
    };
    let Some((count_node, _, _)) = args.get(1) else {
        return Type::Deferred;
    };
    Type::Measure {
        domain: Box::new(Type::Array {
            shape: count_dims(inf, *count_node),
            elem: Box::new(domain),
        }),
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

/// The dims of an `iid` count argument: a vector literal contributes one dim
/// per element, anything else a single dim (see [`resolve_dim`]).
fn count_dims(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Box<[Dim]> {
    if let Node::Call(c) = inf.module.node(node)
        && matches!(c.head, flatppl_core::CallHead::Builtin(op)
            if inf.module.resolve(op) == "vector")
    {
        let elements: Vec<NodeId> = c.args.to_vec();
        return elements.iter().map(|&a| resolve_dim(inf, a)).collect();
    }
    Box::new([resolve_dim(inf, node)])
}

/// A single shape dim: literal integers are static at every level; at
/// `Level::Shape` the demand-driven fixed-integer resolver kicks in.
fn resolve_dim(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Dim {
    if let Node::Lit(Scalar::Int(n)) = inf.module.node(node) {
        return static_dim(*n);
    }
    if inf.level >= Level::Shape {
        if let Some(n) = resolve_fixed_int(inf, node, 0) {
            return static_dim(n);
        }
    }
    Dim::Dynamic
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

/// Demand-driven const-eval of a fixed-phase integer expression at a shape
/// position (engine-concepts §17.1, first slice: integers only). "Resolve,
/// don't rewrite" — the IR is read, never modified. Shape observers
/// (`lengthof`) short-circuit off the inferred type instead of recursing
/// into the value, so deferred-by-design computation stays deferred.
/// `None` means not statically resolvable — a non-fixed ancestor, or a
/// value op outside this resolver's reach (legitimately `%dynamic` for now;
/// the op-gap-vs-dynamic distinction hardens with the full §17.1 slice).
fn resolve_fixed_int(inf: &mut Inferencer<'_, '_>, node: NodeId, depth: u32) -> Option<i64> {
    if depth > 64 {
        return None;
    }
    match inf.module.node(node).clone() {
        Node::Lit(Scalar::Int(n)) => Some(n),
        Node::Ref(r) if r.ns == flatppl_core::RefNs::SelfMod => {
            let binding = inf.module.binding_by_name(r.name)?;
            let rhs = inf.module.binding(binding).rhs;
            let (_, phase) = inf.infer_node(rhs);
            if phase == Phase::Fixed {
                resolve_fixed_int(inf, rhs, depth + 1)
            } else {
                None
            }
        }
        Node::Call(c) => {
            let flatppl_core::CallHead::Builtin(op) = c.head else {
                return None;
            };
            let name = inf.module.resolve(op).to_string();
            match name.as_str() {
                // Shape observer: read the inferred dim, never the value.
                "lengthof" | "length" => {
                    let arg = *c.args.first()?;
                    match inf.infer_node(arg).0 {
                        Type::Array { shape, .. } if shape.len() == 1 => match shape[0] {
                            Dim::Static(n) => Some(i64::from(n)),
                            Dim::Dynamic => None,
                        },
                        Type::TVector {
                            len: Dim::Static(n),
                            ..
                        } => Some(i64::from(n)),
                        _ => None,
                    }
                }
                "add" | "sub" | "mul" => {
                    let a = resolve_fixed_int(inf, *c.args.first()?, depth + 1)?;
                    let b = resolve_fixed_int(inf, *c.args.get(1)?, depth + 1)?;
                    match name.as_str() {
                        "add" => a.checked_add(b),
                        "sub" => a.checked_sub(b),
                        _ => a.checked_mul(b),
                    }
                }
                "neg" => resolve_fixed_int(inf, *c.args.first()?, depth + 1).map(|n| -n),
                _ => None,
            }
        }
        _ => None,
    }
}

fn static_dim(n: i64) -> Dim {
    u32::try_from(n).map(Dim::Static).unwrap_or(Dim::Dynamic)
}

/// `joint(a = M1, b = M2, …)` — a measure over the record of the components'
/// domains (the positional form is deferred with the shape work).
fn joint_type(args: &[ArgInfo], named: &[NamedInfo]) -> Type {
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
    // component variates (spec §06) — same shape-class rule as positional
    // `cartprod` and `cat`: all scalars → a vector, all vectors → a
    // concatenated vector, all records → a merged record; a mixed shape class
    // defers. (Not a record-per-component — that is the keyword form above.)
    let mut domains = Vec::with_capacity(args.len());
    for (_, t, _) in args {
        match t {
            Type::Measure { domain, .. } => domains.push(domain.as_ref().clone()),
            _ => return Type::Deferred,
        }
    }
    Type::Measure {
        domain: Box::new(cat_compose(&domains)),
        mass: Mass::Deferred,
    }
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
    let inputs: Box<[Symbol]> = match call.inputs.as_ref() {
        Some(Inputs::Spec(entries)) => entries.iter().map(|(n, _)| *n).collect(),
        Some(Inputs::Auto) => match inf.module.auto_inputs_of(id) {
            Some(entries) => entries.iter().map(|(n, _)| *n).collect(),
            None => {
                // §04 auto-trace: discover the body's `elementof` parametric
                // leaves (canonical-sorted by name) and fill the side-table, so
                // the reification types as a kernel/function over those inputs.
                let Some((body, _, _)) = args.first() else {
                    return Type::Deferred;
                };
                let entries = inf.collect_auto_inputs(*body);
                let names: Box<[Symbol]> = entries.iter().map(|(n, _)| *n).collect();
                inf.module.set_auto_inputs(id, entries.into());
                names
            }
        },
        None => unreachable!("reification_type called only when inputs are present"),
    };
    let body_ty = args.first().map(|(_, t, _)| t);
    match (name, body_ty) {
        // `kernelof` reifies the LAW of a value-typed body — a probability
        // measure per input, i.e. a Markov kernel.
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
fn cat_compose(types: &[Type]) -> Type {
    let Some(first) = types.first() else {
        return Type::Deferred;
    };
    if types.iter().any(|t| matches!(t, Type::Deferred)) {
        return Type::Deferred;
    }
    match first {
        // all scalars → a length-n vector
        Type::Scalar(_) => {
            let mut elem: Option<Type> = None;
            for t in types {
                if !matches!(t, Type::Scalar(_)) {
                    return Type::Deferred; // mixed shape class
                }
                elem = Some(match elem {
                    None => t.clone(),
                    Some(prev) if &prev == t => prev,
                    Some(prev) => match promote2(Some(&prev), Some(t)) {
                        Type::Deferred => return Type::Deferred,
                        p => p,
                    },
                });
            }
            Type::Array {
                shape: Box::new([Dim::Static(types.len() as u32)]),
                elem: Box::new(elem.unwrap_or(Type::Any)),
            }
        }
        // all 1-D arrays → one concatenated 1-D array
        Type::Array { shape, .. } if shape.len() == 1 => {
            let mut total = 0u32;
            let mut dynamic = false;
            let mut elem: Option<Type> = None;
            for t in types {
                let Type::Array { shape, elem: e } = t else {
                    return Type::Deferred; // mixed shape class
                };
                if shape.len() != 1 {
                    return Type::Deferred; // higher-rank cat is not a §06 joint
                }
                match shape[0] {
                    Dim::Static(n) => total += n,
                    Dim::Dynamic => dynamic = true,
                }
                elem = Some(match elem {
                    None => (**e).clone(),
                    Some(prev) if &prev == e.as_ref() => prev,
                    Some(prev) => match promote2(Some(&prev), Some(e.as_ref())) {
                        Type::Deferred => return Type::Deferred,
                        p => p,
                    },
                });
            }
            Type::Array {
                shape: Box::new([if dynamic {
                    Dim::Dynamic
                } else {
                    Dim::Static(total)
                }]),
                elem: Box::new(elem.unwrap_or(Type::Any)),
            }
        }
        // all records → a merged record (component fields assumed distinct)
        Type::Record(_) => {
            let mut fields: Vec<(Symbol, Type)> = Vec::new();
            for t in types {
                let Type::Record(fs) = t else {
                    return Type::Deferred; // mixed shape class
                };
                fields.extend(fs.iter().cloned());
            }
            Type::Record(fields.into())
        }
        _ => Type::Deferred,
    }
}

/// Do these components have GENUINELY different recognized shape classes
/// (scalar / 1-D vector / record), with none deferred? Distinguishes a
/// `cat_compose` deferral that is really a spec-§06 "mixing shape classes is a
/// static error" from one that is only "a component isn't inferred yet" — so a
/// caller (positional `cartprod` / `joint`) can raise a loud diagnostic without
/// firing on an unresolved or unclassifiable component.
fn cat_is_mixed(types: &[Type]) -> bool {
    if types.len() < 2 || types.iter().any(|t| matches!(t, Type::Deferred)) {
        return false;
    }
    let class = |t: &Type| match t {
        Type::Scalar(_) => Some(0u8),
        Type::Array { shape, .. } if shape.len() == 1 => Some(1),
        Type::Record(_) => Some(2),
        _ => None,
    };
    let classes: Vec<Option<u8>> = types.iter().map(class).collect();
    if classes.iter().any(Option::is_none) {
        return false; // an unclassifiable component — don't fabricate a "mixed" error
    }
    classes.iter().any(|c| *c != classes[0])
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
        Type::Kernel { mass, .. } => match substituted_result(inf, callee, args, named)
            .map(|(ty, _)| ty)
            .or_else(|| reified_result_type(inf, callee))
        {
            Some(Type::Measure { domain, .. }) => Type::Measure {
                domain,
                mass: *mass,
            },
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
    // `SameScalarKind` / `DomainMap` row correct).
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

/// The result type of a per-name function declared in the catalogue as
/// `Sig::Function`, or `None` if the name is not a known function (so the
/// caller can fall through to distribution dispatch, then gap).
///
/// `arg_scalar` is built from the inferred positional argument types so that
/// `SameScalarKind` and `DomainMap` sigs can read the call-site scalar kind.
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
        // `load_data` is a dynamic-length vector whose entries lie in the
        // declared `valueset`: `CartPow(valueset, %dynamic)`.
        "load_data" => {
            let vs = named_or_positional_node(inf.module, named, args, "valueset", 1);
            match set_expr_valueset(inf, vs) {
                ValueSet::Unknown => ValueSet::Unknown,
                set => ValueSet::CartPow(Box::new(set), Dim::Dynamic),
            }
        }
        // Measure supports (the measure node's value set IS its support).
        "Lebesgue" | "Counting" => set_expr_valueset(inf, args.first().map(|a| a.0)),
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
            let elem = set_expr_valueset(inf, c.args.first().copied());
            if elem == ValueSet::Unknown {
                return ValueSet::Unknown;
            }
            let shape = c.args.get(1).map_or_else(
                || Box::new([Dim::Dynamic]) as Box<[Dim]>,
                |&n| count_dims(inf, n),
            );
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
        // Every §08 distribution is a probability measure.
        "lawof" => Mass::Normalized,
        // `Dirac(value)` is a point-mass probability measure (total mass 1).
        "Dirac" => Mass::Normalized,
        // Reference measures: finite on a bounded support, infinite (but
        // boundedly finite) on an unbounded one.
        "Lebesgue" | "Counting" => {
            match set_expr_valueset(inf, args.first().map(|a| a.0)).is_bounded() {
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
        "joint" => {
            let masses: Vec<Mass> = named
                .iter()
                .map(|(_, _, t, _)| match t {
                    Type::Measure { mass, .. } => *mass,
                    _ => Mass::Unknown,
                })
                .collect();
            product_mass(&masses)
        }
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

/// The mass of an independent product of components.
fn product_mass(masses: &[Mass]) -> Mass {
    use Mass::*;
    if masses.contains(&Null) {
        return Null;
    }
    if masses.iter().all(|m| *m == Normalized) {
        return Normalized;
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
        "LogNormal" | "Gamma" | "InverseGamma" | "ChiSquared" | "Pareto" => PosReals,
        "Exponential" | "Weibull" => NonNegReals,
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
