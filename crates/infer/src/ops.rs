//! The per-op type/phase rule catalogue (engine-concepts §18: one source of
//! truth per op — this table is what later passes share).
//!
//! Coverage is incremental and honest: ops without a rule yield `%deferred`
//! plus a once-per-op note (see crate docs). Rules mirror the spec tables —
//! §07 functions (domains/results), §08 distributions (variate domains),
//! §06 measure combinators, §04 reified callables.

use std::collections::{HashMap, HashSet};

use flatppl_core::{
    Call, CallHead, Dim, Idx, Inputs, Mass, Node, NodeId, Phase, Ref, RefNs, Scalar, ScalarType,
    Symbol, Type, ValueSet, Variance,
};

use crate::Level;
use crate::catalogue::ParamShape;
use crate::consteval::{count_dims, count_dims_with_zero, resolve_count, resolve_dim, static_dim};
use crate::rule::RuleStep::Ready;
use crate::rule::{LocalTargets, RuleStep};
use crate::trace::{Inferencer, join_phase};

mod pyhf;
mod tilt;

/// `(node, type, phase)` of an inferred positional argument.
pub(crate) type ArgInfo = (NodeId, Type, Phase);
/// `(name, node, type, phase)` of an inferred named argument.
pub(crate) type NamedInfo = (Symbol, NodeId, Type, Phase);

/// Maximum array rank inference materializes from a compact count expression.
/// This is an implementation resource guard, not a FlatPPL language limit.
const MAX_STRUCTURAL_RANK: usize = 64;

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
///
/// A failed argument fails the call. No row has a meaning for an operand with no type,
/// and a nominal result would hide the refusal under `%deferred` or a value. The op
/// rule still runs, so its own diagnostics are reported. A reification's arguments are
/// its body and boundary, not operands, so it is exempt.
pub(crate) fn call_rule(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    call: &Call,
    callee: Option<(NodeId, Type)>,
    args: &[ArgInfo],
    named: &[NamedInfo],
    joined: Phase,
) -> RuleStep<(Type, Phase)> {
    let failed = if call.inputs.is_none() {
        args.iter()
            .map(|(_, ty, _)| ty)
            .chain(named.iter().map(|(_, _, ty, _)| ty))
            .find_map(first_failed)
    } else {
        None
    };
    op_rule(inf, id, call, callee, args, named, joined).map(inf, move |_, (ty, phase)| match failed
    {
        Some(failed) if !matches!(ty, Type::Failed(_)) => (failed, phase),
        _ => (ty, phase),
    })
}

/// The op rule of a call, before [`call_rule`]'s failed-argument override.
fn op_rule(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    call: &Call,
    callee: Option<(NodeId, Type)>,
    args: &[ArgInfo],
    named: &[NamedInfo],
    joined: Phase,
) -> RuleStep<(Type, Phase)> {
    // The additional arguments of an applied `fchain`/`kchain`/`jointchain` follow
    // the leading argument in either spelling (§04 "Additional arguments").
    let applied_chain = callee.as_ref().is_some_and(|(node, _)| {
        let resolved = resolve_binding_refs(inf, *node);
        matches!(inf.module.node(resolved), Node::Call(c)
            if matches!(c.head, CallHead::Builtin(op)
                if matches!(inf.module.resolve(op), "fchain" | "kchain" | "jointchain")))
    });
    if !args.is_empty()
        && !named.is_empty()
        && !applied_chain
        && !matches!(call.head, CallHead::Builtin(op) if matches!(inf.module.resolve(op), "functionof" | "kernelof" | "broadcast" | "ksuperpose" | "load_module" | "weighted" | "logweighted" | "pushfwd" | "reduce" | "filter" | "scan" | "kscan" | "markovchain"))
    {
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            "ordinary calls use either positional or keyword arguments, not both",
        ));
        return Ready((Type::Failed("mixed ordinary call arguments".into()), joined));
    }
    // User-defined callable application: the result looks through the callee
    // to the reified body (spec §11 reified callables).
    if let Some((callee_node, callee_ty)) = callee {
        let resolved = resolve_binding_refs(inf, callee_node);
        if let Node::Const(op) = inf.module.node(resolved) {
            let mut call = call.clone();
            call.head = CallHead::Builtin(*op);
            return call_rule(inf, id, &call, None, args, named, joined);
        }
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
            return Ready((
                Type::Failed(format!("{name} is not callable").into()),
                Phase::Fixed,
            ));
        }
        if let Some(ty) = pyhf::call_type(inf, id, callee_node, args, named, false) {
            return Ready((ty, joined));
        }
        if let Some(ty) = user_arity_check(inf, id, callee_node, args, named) {
            return Ready((ty, joined));
        }
        return user_call_type(inf, callee_node, &callee_ty, args, named)
            .map(inf, move |_, ty| (ty, joined));
    }

    let flatppl_core::CallHead::Builtin(op) = call.head else {
        unreachable!("user calls handled above");
    };
    let name = inf.module.resolve(op).to_string();
    if matches!(name.as_str(), "load_module" | "load_data")
        && let Some(source) = named_or_positional_node(inf.module, named, args, "source", 0)
        && let Some(value) = written_string(inf, source)
        && let Some(scheme) = flatppl_core::text::uri_scheme(&value)
        && !["file", "http", "https"]
            .iter()
            .any(|s| scheme.eq_ignore_ascii_case(s))
    {
        inf.diags.push(crate::Diagnostic::error_at(
            source,
            "only file, http and https source schemes are allowed (spec §04)",
        ));
        return Ready((
            Type::Failed("unsupported source scheme".into()),
            Phase::Fixed,
        ));
    }

    // Reified callables (`functionof` / `kernelof`) — typed by their boundary
    // + body. The phase follows the CAPTURED ancestors, so the rule returns it
    // rather than assuming `%fixed`; see [`reification_type`].
    if call.inputs.is_some() {
        return Ready(reification_type(inf, id, call, &name, args));
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
        return Ready((ty, joined));
    }

    // §04's special operations declare their input counts in prose, not in the
    // catalogue, so `arity_check` above returns `None` for every one of them.
    // This is the same rule read off that list, plus §04's flat "Nullary calls
    // (`f()`) are not allowed."
    if let Some(ty) = special_arity_check(inf, id, &name, args, named) {
        return Ready((ty, joined));
    }

    // §03 well-formedness of the aggregate CONSTRUCTORS, before the per-op rules
    // build a type from names they have not checked are usable.
    if let Some(ty) = constructor_check(inf, id, &name, args, named) {
        return Ready((ty, joined));
    }

    // Written sizes, after `arity_check` (which needs the call as written) and
    // ahead of the per-op rules, which read the size to build a shape.
    if let Some(ty) = written_size_check(inf, &name, args, named) {
        return Ready((ty, joined));
    }
    if let Some(ty) = single_point_linspace_check(inf, id, &name, args, named) {
        return Ready((ty, joined));
    }

    // AFTER `arity_check`, which must see the call as WRITTEN: it counts
    // `args.len() + named.len()` and resolves the splat, so handing it a vector holding the
    // keywords as well as `named` would double every keyword. Normalization is for POSITIONS
    // only; counts and names are settled above.
    // §04 "Additional arguments": the callback's additional arguments bind its
    // remaining inputs, in input order, so every rule below sees them positionally.
    let extras_bound;
    let (args, named) = match bind_call_extras(inf, id, &name, args, named) {
        ExtrasOutcome::Unchanged => (args, named),
        ExtrasOutcome::Bound(bound) => {
            extras_bound = bound;
            (extras_bound.as_slice(), &[][..])
        }
        ExtrasOutcome::Refused(ty) => return Ready((ty, joined)),
    };
    let normalized;
    let args = match normalize_keyword_args(inf.module, &name, args, named) {
        Some(v) => {
            normalized = v;
            normalized.as_slice()
        }
        None => args,
    };

    // Argument DOMAINS, after normalization (the checks index declared positions)
    // and ahead of the per-op rules, which read an argument's type without asking
    // whether the row's documented domain admits it.
    if let Some(ty) = domain_check(inf, &name, args, named) {
        return Ready((ty, joined));
    }
    if name == "in"
        && let [(point, actual, _), (set, _, _)] = args
    {
        let expected = set_element_type(inf, Some(*set));
        if membership_scalar_mismatch(&expected, actual) {
            inf.diags.push(crate::Diagnostic::error_at(
                *point,
                format!(
                    "`in` expects a value matching the set's element type {}, got {} (spec §07)",
                    crate::boundary::show(inf, &expected),
                    crate::boundary::show(inf, actual),
                ),
            ));
            return Ready((Type::Failed("membership type mismatch".into()), joined));
        }
    }
    if let Some(ty) = refuse_argument_outside_declared_shape(inf, &name, args, false) {
        return Ready((ty, joined));
    }
    if let Some(ty) = refuse_lossy_literal_conversion(inf, &name, args) {
        return Ready((ty, joined));
    }

    if let Some(ty) = callback_arity_check(inf, id, &name, args) {
        return Ready((ty, joined));
    }

    if matches!(name.as_str(), "weighted" | "logweighted")
        && (args.len() >= 3 || matches!(arg_ty(args, 0), Some(Type::Function { .. })))
    {
        let result_type = fresh_measure(arg_ty(args, 1));
        let Type::Measure { domain, .. } = &result_type else {
            return Ready((result_type, joined));
        };
        let mut inputs = vec![(args[1].0, *domain.clone(), Phase::Parametric)];
        inputs.extend(args.iter().skip(2).cloned());
        let function = args[0].0;
        let callable = matches!(
            args[0].1,
            Type::Function { .. } | Type::Any | Type::Deferred
        );
        let resolved = resolve_binding_refs(inf, function);
        let result = if let Node::Const(op) = inf.module.node(resolved) {
            // Builtin function values use the ordinary call checks too.
            let application = Call {
                head: CallHead::Builtin(*op),
                args: inputs.iter().map(|arg| arg.0).collect(),
                named: Box::new([]),
                inputs: None,
            };
            call_rule(inf, id, &application, None, &inputs, &[], joined)
                .map(inf, |_, (ty, _)| Some(ty))
        } else {
            substituted_result(inf, function, &inputs, &[], None, false)
                .map(inf, |_, result| result.map(|(ty, _)| ty))
        };
        return result.map(inf, move |inf, result| {
            let ty = result.or_else(|| reified_result_type(inf, function));
            if ty.as_ref().is_some_and(|ty| {
                !matches!(
                    ty,
                    Type::Scalar(ScalarType::Real | ScalarType::Integer)
                        | Type::Any
                        | Type::Deferred
                        | Type::Failed(_)
                )
            }) || !callable
            {
                inf.diags.push(crate::Diagnostic::error_at(
                    id,
                    "a weight callback must return a real scalar from the whole variate and optional parameter",
                ));
                return (Type::Failed("invalid weight function".into()), joined);
            }
            (result_type, joined)
        });
    }

    let ty = match name.as_str() {
        flatppl_core::LOG_INTEGRAL
        | flatppl_core::MASS_INTEGRAL
        | flatppl_core::LOG_INTERVAL_MASS
        | flatppl_core::INTEGRATION_POINT => Type::Scalar(ScalarType::Real),
        flatppl_core::SORT_VECTOR => arg_ty(args, 0).cloned().unwrap_or(Type::Deferred),
        // ---- arithmetic (spec §07) — structural: result depends on arg shapes/types ----
        "add" | "sub" => elementwise2(&args.first(), &args.get(1)),
        "mul" => mul_type(args),
        "linsolve" => linsolve_type(inf, id, args),
        "qr" => qr_type(inf, id, args),
        "cross" => cross_type(inf, id, args),
        "pow" => promote2(arg_ty(args, 0), arg_ty(args, 1)),
        "neg" => args.first().map_or(Type::Deferred, |(_, t, _)| t.clone()),
        "min" | "max" | "atan2" => promote2(arg_ty(args, 0), arg_ty(args, 1)),
        // `divide(a, b) = a / b` over real OR complex scalars (spec §07):
        // complex if either operand is complex, else real (true division —
        // integer operands still divide to real). NOT a constant Scalar(Real):
        // the complex case must promote. See `divide_type`.
        "divide" => divide_type(arg_ty(args, 0), arg_ty(args, 1)),

        "div" | "mod" => integer_arithmetic_type(inf, &name, args),

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
        "rowstack" | "colstack" => stack_type(arg_ty(args, 0), name == "colstack"),
        "conv" | "crosscorr" => convolution_type(inf, id, &name, args),
        "get" | "get0" if refuse_real_index(inf, &name, args) => {
            Type::Failed(format!("{name}: real-typed index").into())
        }
        "get" => get_type(inf, args, /*base=*/ 1),
        "get0" => get_type(inf, args, /*base=*/ 0),
        // §07 gives every comparison ("Operator-equivalent functions") and every
        // logical connective ("Logic and conditionals") a SCALAR domain, so a
        // non-scalar operand has no §07 meaning — see [`refuse_nonscalar_operand`],
        // which is where the citations live.
        "equal" | "unequal" | "lt" | "le" | "gt" | "ge" | "land" | "lor" | "lxor" | "lnot"
            if nonscalar_operand(args, named).is_some() =>
        {
            refuse_nonscalar_operand(inf, id, &name, args, named)
        }
        "indicesof" | "indicesof0" => indices_type(arg_ty(args, 0)),
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
        "eye" => {
            let n = args.first().map_or(Dim::Dynamic, |a| resolve_dim(inf, a.0));
            Type::Array {
                shape: Box::new([n, n]),
                elem: Box::new(Type::Scalar(ScalarType::Real)),
            }
        }
        "sizeof" => sizeof_type(arg_ty(args, 0)),
        "zeros" | "ones" => Type::Array {
            shape: args.first().map_or_else(
                || Box::new([Dim::Dynamic]) as Box<[Dim]>,
                |a| count_dims(inf, a.0),
            ),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        },
        "onehot" => Type::Array {
            shape: Box::new([args.get(1).map_or(Dim::Dynamic, |a| resolve_dim(inf, a.0))]),
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
            if name == "metricsum" {
                // First: every check below reads `output_axes` as a literal list.
                if let Some(failed) = metricsum_output_axes_check(inf, args) {
                    return Ready((failed, joined));
                }
                if let Some(failed) = metricsum_neutral_axis_check(inf, args) {
                    return Ready((failed, joined));
                }
                if let Some(failed) = metricsum_metric_check(inf, args) {
                    return Ready((failed, joined));
                }
                if let Some(failed) = metricsum_expr_check(inf, args) {
                    return Ready((failed, joined));
                }
                if let Some(failed) = metricsum_output_index_check(inf, args) {
                    return Ready((failed, joined));
                }
                if let Some(failed) = metricsum_contraction_check(inf, args) {
                    return Ready((failed, joined));
                }
            }
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
                        collect_axis_dims(inf, b.0, &axes, &mut extents);
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
            Some(Type::Table { columns, .. }) => Type::Record(columns.clone()),
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
            // The counts are non-negative fixed integers, so they resolve as
            // COUNTS, not sizes: `resolve_count` keeps a legal `Static(0)`
            // where `resolve_dim` floors a size positive. It also emits the
            // loud op-gap diagnostic if a count uses an unfoldable op (§17.1).
            let nl = args.get(1).map(|a| resolve_count(inf, a.0));
            let nt = args.get(2).map(|a| resolve_count(inf, a.0));
            match (arg_ty(args, 0), nl, nt) {
                (
                    Some(Type::Array { shape, elem }),
                    Some(Dim::Static(nl)),
                    Some(Dim::Static(nt)),
                ) => {
                    let rank = (nl as usize)
                        .checked_add(shape.len())
                        .and_then(|rank| rank.checked_add(nt as usize));
                    let Some(rank) = rank.filter(|&rank| rank <= MAX_STRUCTURAL_RANK) else {
                        inf.diags.push(crate::Diagnostic::error_at(
                            id,
                            format!(
                                "`addaxes` result structural rank exceeds the implementation \
                                 resource guard of {MAX_STRUCTURAL_RANK}; reduce `n_leading` or \
                                 `n_trailing`"
                            ),
                        ));
                        return Ready((
                            Type::Failed("addaxes structural rank exceeds resource guard".into()),
                            joined,
                        ));
                    };
                    let mut dims = Vec::with_capacity(rank);
                    dims.extend(std::iter::repeat_n(static_dim(1), nl as usize));
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
        "joinblocks" => match arg_ty(args, 0) {
            Some(Type::Array { shape: outer, elem }) => match elem.as_ref() {
                Type::Array { shape: inner, elem } if outer.len() == inner.len() => {
                    let shape = outer
                        .iter()
                        .zip(inner.iter())
                        .map(|(a, b)| match (a, b) {
                            (Dim::Static(a), Dim::Static(b)) => {
                                a.checked_mul(*b).map_or(Dim::Dynamic, Dim::Static)
                            }
                            _ => Dim::Dynamic,
                        })
                        .collect();
                    Type::Array {
                        shape,
                        elem: elem.clone(),
                    }
                }
                Type::Deferred | Type::Any => Type::Deferred,
                _ => {
                    inf.diags.push(crate::Diagnostic::error_at(id, "`joinblocks` requires an array of arrays with equal outer and inner ranks (spec §07)"));
                    Type::Failed("joinblocks: invalid array nesting".into())
                }
            },
            Some(Type::Deferred | Type::Any) | None => Type::Deferred,
            Some(_) => {
                inf.diags.push(crate::Diagnostic::error_at(id, "`joinblocks` requires an array of arrays with equal outer and inner ranks (spec §07)"));
                Type::Failed("joinblocks: invalid array nesting".into())
            }
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
        "lawof" => match lawof_value_gate(inf, args) {
            Some(failed) => failed,
            None => lawof_type(args.first().map(|(_, t, _)| t)),
        },
        "draw" => match draw_mass_gate(inf, args) {
            Some(failed) => failed,
            None => measure_domain(arg_ty(args, 0)),
        },
        "iid" => iid_type(inf, args),
        "PoissonProcess" => poisson_process_type(inf, args, named),
        "BinnedPoissonProcess" => binned_poisson_process_type(inf, &name, args, named),
        // Measure-transforming ops keep the domain but get a FRESH mass slot
        // — their total mass differs from the base's and is computed by the
        // normalization-level rules (inheriting it via the type clone would
        // smuggle the base's class through `fill_mass`).
        "truncate" | "normalize" => fresh_measure(arg_ty(args, 0)),
        "tilt" => tilt::call_type(inf, id, args),
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
        "ksuperpose" => {
            return ksuperpose_type(inf, id, args, named).map(inf, move |_, ty| (ty, joined));
        }
        // `pushfwd(f, M)` (spec §06): a measure whose domain is the CODOMAIN of
        // `f`. `f` maps a value drawn from `M`, so binding its input to `M`'s
        // variate (domain + support value-set) and reading `f`'s body type gives
        // the codomain; fall back to `f`'s un-substituted body, then to `%any`
        // (honest — never a guess). Mass is filled (mass-preserving) by `fill_mass`.
        "pushfwd" => {
            return pushfwd_codomain(inf, args).map(inf, move |_, codomain| {
                (
                    Type::Measure {
                        domain: Box::new(codomain.unwrap_or(Type::Any)),
                        mass: Mass::Deferred,
                    },
                    joined,
                )
            });
        }
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
            // The sequences after `init` are traversed together (§04 "Additional
            // arguments"), so the trajectory has as many steps as they agree on.
            let lengths: Option<Vec<Option<Dim>>> = args[2.min(args.len())..]
                .iter()
                .map(|input| sequence_item(input).map(|(len, _)| len))
                .collect();
            let len = lengths
                .map(|lengths| lengths.into_iter().flatten().collect::<Vec<_>>())
                .filter(|lengths| !lengths.is_empty())
                .map_or(Dim::Dynamic, |lengths| sequence_steps(&lengths));
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
        "scan" => return scan_type(inf, args).map(inf, move |_, ty| (ty, joined)),
        // `fchain(f1, f2, …)` (spec §04) composes deterministic functions; the
        // result is a function with `f1`'s input signature (output type is not
        // tracked by `Type::Function`).
        "fchain" => args
            .first()
            .and_then(|(node, ..)| crate::callable::interface(inf, *node))
            .map(|interface| Type::Function {
                inputs: interface
                    .names
                    .iter()
                    .map(|name| inf.module.intern(name))
                    .collect(),
            })
            .unwrap_or(Type::Deferred),
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
        // Unary reference measures take their support set whole.
        "Lebesgue" | "Counting" => {
            let support_node = lebesgue_counting_support_node(inf, args, named);
            Type::Measure {
                domain: Box::new(set_element_type(inf, support_node)),
                mass: Mass::Deferred,
            }
        }
        // `Dirac(value = v)` (spec §06) is the point-mass probability measure at
        // `v`, for any variate type: the domain is `v`'s type. A positional
        // record is the point itself, just as a keyword-bound record is. Mass
        // is normalized (total mass 1) — set in `fill_mass`.
        "Dirac" => {
            let v = named
                .iter()
                .find(|(n, _, _, _)| inf.module.resolve(*n) == "value")
                .map(|(_, _, t, _)| t.clone())
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
            None => match measure_domain(arg_ty(args, 0)) {
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
        "interval" | "cartprod" | "stdsimplex" | "ordered" | "zerosum" => Type::Any,
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
        "broadcast" => {
            // A failed cell fails the broadcast, as a failed argument fails a call.
            return broadcast_type(inf, id, args, named)
                .map(inf, move |_, ty| (first_failed(&ty).unwrap_or(ty), joined));
        }
        // Like a bare builtin callable, the lifted callable's result depends
        // on application arguments. Kernel primitives inspect it structurally.
        "broadcasted" => arg_ty(args, 0).cloned().unwrap_or(Type::Deferred),

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
                        return Ready((Type::Failed("unresolvable name".into()), Phase::Fixed));
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
                        return Ready((
                            Type::Failed(format!("{name} is not callable").into()),
                            Phase::Fixed,
                        ));
                    }
                    inf.note_gap(op);
                    Type::Deferred
                }
            },
        },
    };

    let phase = match name.as_str() {
        // A bound coordinate is internal to the integral, not a host parameter.
        // Its value set stays unknown, so constant evaluation cannot read it.
        flatppl_core::INTEGRATION_POINT => Phase::Fixed,
        "elementof" => Phase::Parametric,
        "external" | "load_data" | "load_module" | "standard_module" => Phase::Fixed,
        "draw" => Phase::Stochastic,
        // `lawof` reifies a value into its law; the law is deterministic
        // (parametric or fixed) — `lawof` absorbs the stochasticity of the
        // `draw` ancestors rather than propagating it (spec §04 "Phase of the
        // reified law"). Trace the argument's law-phase instead of inheriting
        // the stochastic `joined`.
        "lawof" => args
            .first()
            .map_or(Phase::Fixed, |a| law_phase(inf, a.0, 0)),
        _ => joined,
    };
    Ready((ty, phase))
}

fn membership_scalar_mismatch(expected: &Type, actual: &Type) -> bool {
    fn rank(ty: &Type) -> Option<u8> {
        match ty {
            Type::Array { elem, .. } => rank(elem),
            Type::Scalar(ScalarType::Boolean) => Some(0),
            Type::Scalar(ScalarType::Integer) => Some(1),
            Type::Scalar(ScalarType::Real) => Some(2),
            Type::Scalar(ScalarType::Complex) => Some(3),
            _ => None,
        }
    }
    matches!((rank(expected), rank(actual)), (Some(want), Some(have)) if have > want)
}

/// How an operation's additional arguments reach its callback (§04 "Additional
/// arguments", §05 table of call forms).
struct ExtrasHead {
    /// Arguments the operation takes before the additional ones.
    lead: usize,
    /// Inputs the operation itself supplies to the callback first.
    callback_lead: usize,
    /// Whether the additional arguments are sequences traversed together
    /// (`filter`, `scan`, `kscan`) rather than values held constant.
    sequences: bool,
}

fn extras_head(name: &str) -> Option<ExtrasHead> {
    let (lead, callback_lead, sequences) = match name {
        "weighted" | "logweighted" | "pushfwd" => (2, 1, false),
        "reduce" => (2, 2, false),
        "markovchain" => (3, 1, false),
        "filter" | "scan" | "kscan" => (2, 1, true),
        _ => return None,
    };
    Some(ExtrasHead {
        lead,
        callback_lead,
        sequences,
    })
}

enum ExtrasOutcome {
    Unchanged,
    /// The call's arguments in callback-input order, no keywords left.
    Bound(Vec<ArgInfo>),
    Refused(Type),
}

/// Bind the additional arguments of `name`'s callback with [`crate::extras::bind`].
fn bind_call_extras(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> ExtrasOutcome {
    let Some(head) = extras_head(name) else {
        return ExtrasOutcome::Unchanged;
    };
    if args.len() < head.lead {
        return ExtrasOutcome::Unchanged;
    }
    let extras = &args[head.lead..];
    if extras.is_empty() && named.is_empty() {
        if head.sequences && name != "filter" {
            return refuse_extras(
                inf,
                id,
                name,
                "needs at least one array or table input to traverse".to_string(),
            );
        }
        return ExtrasOutcome::Unchanged;
    }
    let Some(interface) = crate::callable::interface(inf, args[0].0) else {
        // A callback whose inputs are unknown cannot take keyword arguments; positional
        // ones keep their order, which is all the later rules read.
        return if named.is_empty() {
            ExtrasOutcome::Unchanged
        } else {
            refuse_extras(
                inf,
                id,
                name,
                "the callback's inputs are not known, so keyword arguments cannot bind".to_string(),
            )
        };
    };
    let bound_extras = match bind_extras_to_interface(
        inf,
        id,
        name,
        args[0].0,
        &interface,
        head.callback_lead,
        extras,
        named,
    ) {
        Ok(bound) => bound,
        Err(refusal) => return ExtrasOutcome::Refused(refusal),
    };
    let mut bound: Vec<ArgInfo> = args[..head.lead].to_vec();
    bound.extend(bound_extras);
    // `filter` traverses its additional inputs with `data`, which is a sequence
    // too; `scan` and `kscan` traverse the additional inputs alone.
    let traversed = if name == "filter" { 1 } else { head.lead };
    if name == "filter"
        && let Some((node, ty, _)) = bound.get(1)
        && !matches!(
            ty,
            Type::Array { .. }
                | Type::Table { .. }
                | Type::Deferred
                | Type::Any
                | Type::Var(_)
                | Type::Failed(_)
        )
    {
        return refuse_extras(
            inf,
            *node,
            name,
            "the data to filter must be an array or table".to_string(),
        );
    }
    if head.sequences
        && let Some(refusal) = check_sequences(inf, id, name, &bound[traversed..])
    {
        return ExtrasOutcome::Refused(refusal);
    }
    ExtrasOutcome::Bound(bound)
}

fn refuse_extras(
    inf: &mut Inferencer<'_, '_>,
    node: NodeId,
    name: &str,
    what: String,
) -> ExtrasOutcome {
    ExtrasOutcome::Refused(extras_diagnostic(inf, node, name, what))
}

fn extras_diagnostic(inf: &mut Inferencer<'_, '_>, node: NodeId, name: &str, what: String) -> Type {
    inf.diags.push(crate::Diagnostic::error_at(
        node,
        format!("`{name}`: {what} (spec §04 \"Additional arguments\")"),
    ));
    Type::Failed(format!("{name}: additional arguments").into())
}

/// The additional arguments `extras` and `named` of a call to a callback with
/// `interface`, in the order of the callback's inputs after its first
/// `callback_lead` ones. `who` names the operation in a diagnostic.
#[allow(clippy::too_many_arguments)]
fn bind_extras_to_interface(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    who: &str,
    callback: NodeId,
    interface: &crate::callable::Interface,
    callback_lead: usize,
    extras: &[ArgInfo],
    named: &[NamedInfo],
) -> Result<Vec<ArgInfo>, Type> {
    use crate::extras::{Error, Remaining, Source, bind};
    if interface.names.len() < callback_lead {
        return Err(extras_diagnostic(
            inf,
            callback,
            who,
            format!(
                "the callback declares {} input(s) but is called with {callback_lead} before any additional argument",
                interface.names.len(),
            ),
        ));
    }
    let remaining_names = interface.names[callback_lead..].to_vec();
    let remaining = Remaining {
        names: &remaining_names,
        ordered: interface.ordered,
    };
    // Sequence callbacks receive record parameters whole; table columns retain
    // their existing binding to the callback's remaining inputs.
    let aggregate_fields = match extras {
        [(_, Type::Record(fields), _)] if !matches!(who, "scan" | "kscan" | "filter") => {
            Some(fields.as_ref())
        }
        [(_, Type::Table { columns, .. }, _)] => Some(columns.as_ref()),
        _ => None,
    }
    .filter(|_| named.is_empty());
    let aggregate_names: Option<Vec<String>> = aggregate_fields.map(|fields| {
        fields
            .iter()
            .map(|(n, _)| inf.module.resolve(*n).to_string())
            .collect()
    });
    let named_names: Vec<String> = named
        .iter()
        .map(|(n, ..)| inf.module.resolve(*n).to_string())
        .collect();
    let named_refs: Vec<&str> = named_names.iter().map(String::as_str).collect();
    let sources = bind(
        &remaining,
        extras.len(),
        &named_refs,
        aggregate_names.as_deref(),
    )
    .map_err(|error| {
        let what = match error {
            Error::KeywordOnly => {
                "the callback has keyword-only inputs, so additional arguments bind by name"
                    .to_string()
            }
            Error::TooMany { given, inputs } => format!(
                "{given} additional argument(s) but the callback has {inputs} remaining input(s)"
            ),
            Error::Unknown(n) => format!("the callback has no remaining input `{n}`"),
            Error::Twice(n) => format!("input `{n}` is bound twice"),
            Error::Unbound(n) => format!("input `{n}` is left unbound"),
        };
        extras_diagnostic(inf, id, who, what)
    })?;
    // The sole additional record or table splats by name: its fields read through
    // the same projection the ordinary auto-splat uses.
    let splat =
        if aggregate_fields.is_some() && extras.len() == 1 && !remaining.accepts_one_positional() {
            normalize_arguments(inf.module, &remaining_names, extras, &[], false, true)
        } else {
            None
        };
    let mut bound = Vec::with_capacity(sources.len());
    for (slot, source) in sources.iter().enumerate() {
        bound.push(match source {
            Source::Positional(k) => extras[*k].clone(),
            Source::Named(j) => {
                let (_, node, ty, phase) = &named[*j];
                (*node, ty.clone(), *phase)
            }
            Source::Field(_) => match &splat {
                Some(fields) => fields[slot].clone(),
                // The field's type is not readable, so the call stays as written.
                None => return Ok(extras.to_vec()),
            },
        });
    }
    Ok(bound)
}

/// §04 "Additional arguments": the additional inputs of `filter`, `scan` and `kscan`
/// traverse one-dimensional arrays and tables together. A length-one input repeats;
/// other unequal lengths are an error. Scalars, records and other non-collection
/// inputs are held constant. A tuple or an array of more than one dimension is an
/// error, and `scan` and `kscan` need at least one array or table to traverse.
fn check_sequences(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    inputs: &[ArgInfo],
) -> Option<Type> {
    let mut len: Option<u32> = None;
    let mut traversed = false;
    let mut unknown = false;
    for (node, ty, _) in inputs {
        let dim = match ty {
            Type::Array { shape, .. } if shape.len() == 1 => shape[0],
            Type::Table { nrows, .. } => *nrows,
            Type::Array { .. } | Type::Tuple(_) => {
                inf.diags.push(crate::Diagnostic::error_at(
                    *node,
                    format!(
                        "`{name}` additional inputs are one-dimensional arrays or tables, or \
                         constants; a tuple or a higher-dimensional array is not allowed \
                         (spec §04 \"Additional arguments\")"
                    ),
                ));
                return Some(Type::Failed(
                    format!("{name}: input is not a sequence or a constant").into(),
                ));
            }
            Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_) => {
                unknown = true;
                continue;
            }
            _ => continue,
        };
        traversed = true;
        let Dim::Static(n) = dim else { continue };
        if n == 1 {
            continue;
        }
        match len {
            Some(previous) if previous != n => {
                inf.diags.push(crate::Diagnostic::error_at(
                    *node,
                    format!(
                        "`{name}` sequence inputs have lengths {previous} and {n}: spec §04 \
                         \"Additional arguments\" repeats only length-one inputs, and other \
                         unequal lengths are an error"
                    ),
                ));
                return Some(Type::Failed(
                    format!("{name}: unequal sequence lengths").into(),
                ));
            }
            _ => len = Some(n),
        }
    }
    if name != "filter" && !traversed && !unknown {
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            format!(
                "`{name}` needs at least one array or table input to traverse; a constant is \
                 held at every step (spec §04 \"Additional arguments\")"
            ),
        ));
        return Some(Type::Failed(format!("{name}: no sequence input").into()));
    }
    None
}

/// Higher-order operations pass whole objects in a fixed callback signature.
fn callback_arity_check(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    op: &str,
    args: &[ArgInfo],
) -> Option<Type> {
    let head = extras_head(op)?;
    let callback = args.first()?;
    // The operation supplies `callback_lead` inputs itself, and each additional
    // argument binds one more.
    let expected = head.callback_lead + args.len().saturating_sub(head.lead);
    let count = match &callback.1 {
        Type::Function { inputs } | Type::Kernel { inputs, .. } => Some(inputs.len()),
        _ => {
            let node = resolve_binding_refs(inf, callback.0);
            match inf.module.node(node) {
                Node::Const(name) => {
                    let arity =
                        crate::catalogue::builtin().base_arity(inf.module.resolve(*name))?;
                    if arity.admits(expected) {
                        return None;
                    }
                    Some(arity.min)
                }
                _ => None,
            }
        }
    }?;
    if count == expected {
        return None;
    }
    inf.diags.push(crate::Diagnostic::error_at(id, format!(
        "`{op}` calls its callback with {expected} whole argument(s), but the callback declares {count}"
    )));
    Some(Type::Failed("callback arity mismatch".into()))
}

fn arg_ty(args: &[ArgInfo], i: usize) -> Option<&Type> {
    args.get(i).map(|(_, t, _)| t)
}

/// The number of steps of sequences traversed together (§04 "Additional
/// arguments"): equal lengths, with length-one inputs repeating. An empty input
/// combined with length-one inputs yields zero steps.
fn sequence_steps(lengths: &[Dim]) -> Dim {
    let mut steps = None;
    let mut dynamic = false;
    for length in lengths {
        match length {
            Dim::Static(1) => {}
            Dim::Static(n) => steps = Some(*n),
            Dim::Dynamic => dynamic = true,
        }
    }
    match (steps, dynamic) {
        (Some(n), _) => Dim::Static(n),
        (None, true) => Dim::Dynamic,
        (None, false) => Dim::Static(1),
    }
}

/// One additional input of `scan`/`kscan`, as the step sees it: `Some(len)` with
/// the element (row) type for a one-dimensional array or table, `None` with the
/// whole value for a constant. A tuple, a higher-dimensional array or an input of
/// unknown type is `None` for the whole call (inference of the step is deferred;
/// [`check_sequences`] reports the tuple and the array).
fn sequence_item(input: &ArgInfo) -> Option<(Option<Dim>, ArgInfo)> {
    let (len, item) = match &input.1 {
        Type::Array { shape, elem } if shape.len() == 1 => (Some(shape[0]), (**elem).clone()),
        Type::Table { columns, nrows } => (Some(*nrows), Type::Record(columns.clone())),
        Type::Array { .. }
        | Type::Tuple(_)
        | Type::Deferred
        | Type::Any
        | Type::Var(_)
        | Type::Failed(_) => return None,
        other => (None, other.clone()),
    };
    Some((len, (input.0, item, input.2)))
}

/// A scan has one fixed state type, including the numeric embedding of an
/// integer initial value into a real-valued recurrence. Infer the step with
/// state and row types, just as broadcast infers its callable on cell types.
/// The sequences after `init` are traversed together, and their elements are
/// the step's inputs after the state (§04 "Additional arguments").
fn scan_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> RuleStep<Type> {
    let [function, init, sequences @ ..] = args else {
        return Ready(Type::Deferred);
    };
    let Some(items) = sequences
        .iter()
        .map(sequence_item)
        .collect::<Option<Vec<_>>>()
        .filter(|items| items.iter().any(|(len, _)| len.is_some()))
    else {
        return Ready(Type::Deferred);
    };
    let len = sequence_steps(&items.iter().filter_map(|(len, _)| *len).collect::<Vec<_>>());
    let items: Vec<ArgInfo> = items.into_iter().map(|(_, item)| item).collect();
    scan_state_type(inf, function.0, init.clone(), items).map(inf, move |_, state| match state {
        Type::Record(columns) => Type::Table {
            columns,
            nrows: len,
        },
        state @ (Type::Scalar(_) | Type::Array { .. }) => Type::Array {
            shape: Box::new([len]),
            elem: Box::new(state),
        },
        _ => Type::Deferred,
    })
}

fn scan_state_type(
    inf: &mut Inferencer<'_, '_>,
    function: NodeId,
    state: ArgInfo,
    items: Vec<ArgInfo>,
) -> RuleStep<Type> {
    let mut arguments = vec![state.clone()];
    arguments.extend(items.iter().cloned());
    substituted_result(inf, function, &arguments, &[], None, false).and_then(
        inf,
        move |inf, result| {
            let result = result
                .map(|(ty, _)| ty)
                .or_else(|| reified_result_type(inf, function));
            let Some(joined) = result
                .as_ref()
                .and_then(|next| scan_state_join(&state.1, next))
            else {
                return Ready(Type::Deferred);
            };
            if joined == state.1 {
                Ready(joined)
            } else {
                // The structure cannot change. Every leaf only widens through
                // the finite boolean/integer/real/complex inclusion chain.
                scan_state_type(inf, function, (state.0, joined, state.2), items)
            }
        },
    )
}

fn scan_state_join(state: &Type, next: &Type) -> Option<Type> {
    if state == next {
        return Some(state.clone());
    }
    match (state, next) {
        (Type::Scalar(_), Type::Scalar(_)) => Some(promote2(Some(state), Some(next))),
        (
            Type::Array { shape, elem },
            Type::Array {
                shape: other,
                elem: next,
            },
        ) if shape == other => Some(Type::Array {
            shape: shape.clone(),
            elem: Box::new(scan_state_join(elem, next)?),
        }),
        (Type::Record(fields), Type::Record(other)) if fields.len() == other.len() => {
            let fields = fields
                .iter()
                .map(|(name, ty)| {
                    let next = &other.iter().find(|(field, _)| field == name)?.1;
                    Some((*name, scan_state_join(ty, next)?))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Type::Record(fields.into()))
        }
        _ => None,
    }
}

/// Bind the ordinary builtin interface before any domain-specific rule.
fn normalize_keyword_args(
    module: &flatppl_core::Module,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Vec<ArgInfo>> {
    let cat = crate::catalogue::builtin();
    normalize_arguments(
        module,
        cat.base_param_names(name)?,
        args,
        named,
        cat.base_takes_aggregate_whole(name),
        true,
    )
}

/// Return arguments in declared order when a keyword or aggregate spelling
/// needs normalization. Plain positional calls keep their original vector.
///
/// A sole positional record or table binds by field name. A sole positional tuple
/// of an `ordered` callable binds by position, each element a positional argument
/// (spec §04 "Calling conventions"). A callable that accepts a lone positional
/// argument (`takes_one`) receives the aggregate whole, so nothing is normalized.
fn normalize_arguments(
    module: &flatppl_core::Module,
    declared: &[String],
    args: &[ArgInfo],
    named: &[NamedInfo],
    takes_one: bool,
    ordered: bool,
) -> Option<Vec<ArgInfo>> {
    let mut slots: Vec<Option<ArgInfo>> = Vec::new();
    if named.is_empty() {
        if takes_one {
            return None;
        }
        let [(node, ty, phase)] = args else {
            return None;
        };
        let mut source = *node;
        for _ in 0..module.node_count() {
            let Node::Ref(r) = module.node(source) else {
                break;
            };
            if r.ns != RefNs::SelfMod {
                break;
            }
            let Some(binding) = module.binding_by_name(r.name) else {
                break;
            };
            source = module.binding(binding).rhs;
        }
        if let Type::Tuple(parts) = ty {
            if !ordered {
                return None;
            }
            // A written `(a, b, …)` binds each element's own node; any other tuple
            // value binds the projection, typed by its element.
            let literal = match module.node(source) {
                Node::Call(call)
                    if matches!(call.head, CallHead::Builtin(head) if module.resolve(head) == "tuple")
                        && call.args.len() == parts.len()
                        && call.named.is_empty() =>
                {
                    Some(&call.args)
                }
                _ => None,
            };
            return Some(
                parts
                    .iter()
                    .enumerate()
                    .map(|(i, part)| {
                        (
                            literal.map_or(*node, |elements| elements[i]),
                            part.clone(),
                            *phase,
                        )
                    })
                    .collect(),
            );
        }
        let (fields, rows) = match ty {
            Type::Record(fields) => (fields.as_ref(), None),
            Type::Table { columns, nrows } => (columns.as_ref(), Some(*nrows)),
            _ => return None,
        };
        for (sym, ty) in fields {
            let pos = declared
                .iter()
                .position(|name| name == module.resolve(*sym))?;
            while slots.len() <= pos {
                slots.push(None);
            }
            let field = match module.node(source) {
                Node::Call(call) if matches!(call.head, CallHead::Builtin(head) if matches!(module.resolve(head), "record" | "table")) => {
                    call.named
                        .iter()
                        .find(|field| field.name == *sym)
                        .map(|field| field.value)
                        .unwrap_or(*node)
                }
                _ => *node,
            };
            let ty = match rows {
                Some(nrows) => Type::Array {
                    shape: vec![nrows].into(),
                    elem: Box::new(ty.clone()),
                },
                None => ty.clone(),
            };
            slots[pos] = Some((field, ty, *phase));
        }
    } else {
        slots.extend(args.iter().cloned().map(Some));
        for (sym, node, ty, phase) in named {
            let pos = declared
                .iter()
                .position(|name| name == module.resolve(*sym))?;
            while slots.len() <= pos {
                slots.push(None);
            }
            if slots[pos].is_some() {
                return None;
            }
            slots[pos] = Some((*node, ty.clone(), *phase));
        }
    }
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

/// §07 `div` and `mod` take integers, including §03's Boolean embedding.
/// Share the ordinary-call check with broadcasting so real/complex array
/// cells and held scalars cannot bypass the integer domain. Unknown kinds
/// stay undecided; real division is the separate `divide` operation.
fn integer_arithmetic_type(inf: &mut Inferencer<'_, '_>, name: &str, args: &[ArgInfo]) -> Type {
    let mut ok = true;
    for (i, (node, t, _)) in args.iter().take(2).enumerate() {
        if let Some(kind @ (ScalarType::Real | ScalarType::Complex)) = scalar_kind(t) {
            let hint = if name == "div" {
                " — use `divide` for real division"
            } else {
                ""
            };
            inf.diags.push(crate::Diagnostic::error_at(
                *node,
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

fn linsolve_type(inf: &mut Inferencer<'_, '_>, id: NodeId, args: &[ArgInfo]) -> Type {
    let Some(Type::Array {
        shape: a_shape,
        elem: a,
    }) = arg_ty(args, 0)
    else {
        return Type::Deferred;
    };
    let (b_dim, b) = match arg_ty(args, 1) {
        Some(Type::Array { shape, elem }) if shape.len() == 1 => (shape[0], elem),
        Some(Type::TVector { len, elem }) => (*len, elem),
        _ => return Type::Deferred,
    };
    if a_shape.len() != 2
        || !matches!(a.as_ref(), Type::Scalar(_))
        || !matches!(b.as_ref(), Type::Scalar(_))
    {
        return Type::Deferred;
    }
    let mismatch = matches!((a_shape[0], a_shape[1]), (Dim::Static(x), Dim::Static(y)) if x != y)
        || matches!((a_shape[1], b_dim), (Dim::Static(x), Dim::Static(y)) if x != y);
    if mismatch {
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            "`linsolve` requires a square matrix and a matching vector (spec §07)",
        ));
        return Type::Failed("linsolve: incompatible shapes".into());
    }
    let kind = if matches!(a.as_ref(), Type::Scalar(ScalarType::Complex))
        || matches!(b.as_ref(), Type::Scalar(ScalarType::Complex))
    {
        ScalarType::Complex
    } else {
        ScalarType::Real
    };
    Type::Array {
        shape: Box::new([b_dim]),
        elem: Box::new(Type::Scalar(kind)),
    }
}

fn qr_type(inf: &mut Inferencer<'_, '_>, id: NodeId, args: &[ArgInfo]) -> Type {
    let Some(Type::Array { shape, elem }) = arg_ty(args, 0) else {
        return Type::Deferred;
    };
    if shape.len() != 2 || !matches!(elem.as_ref(), Type::Scalar(_)) {
        return Type::Deferred;
    }
    let (rows, cols) = (shape[0], shape[1]);
    if matches!((rows, cols), (Dim::Static(m), Dim::Static(n)) if m < n) {
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            "`qr` requires at least as many rows as columns (spec §07)",
        ));
        return Type::Failed("qr: matrix is wider than tall".into());
    }
    let kind = if matches!(elem.as_ref(), Type::Scalar(ScalarType::Complex)) {
        ScalarType::Complex
    } else {
        ScalarType::Real
    };
    let q = Type::Array {
        shape: Box::new([rows, cols]),
        elem: Box::new(Type::Scalar(kind)),
    };
    let r = Type::Array {
        shape: Box::new([cols, cols]),
        elem: Box::new(Type::Scalar(kind)),
    };
    let q_name = inf.module.intern("Q");
    let r_name = inf.module.intern("R");
    Type::Record(Box::new([(q_name, q), (r_name, r)]))
}

fn cross_type(inf: &mut Inferencer<'_, '_>, id: NodeId, args: &[ArgInfo]) -> Type {
    fn vector(ty: &Type) -> Option<(Dim, &Type)> {
        match ty {
            Type::Array { shape, elem } if shape.len() == 1 => Some((shape[0], elem.as_ref())),
            Type::TVector { len, elem } => Some((*len, elem.as_ref())),
            _ => None,
        }
    }
    let (Some((a_len, a)), Some((b_len, b))) = (
        arg_ty(args, 0).and_then(vector),
        arg_ty(args, 1).and_then(vector),
    ) else {
        return Type::Deferred;
    };
    if [a_len, b_len]
        .iter()
        .any(|d| matches!(d, Dim::Static(n) if *n != 3))
    {
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            "`cross` requires two vectors of length 3 (spec §07)",
        ));
        return Type::Failed("cross: vector length must be 3".into());
    }
    Type::Array {
        shape: Box::new([Dim::Static(3)]),
        elem: Box::new(promote2(Some(a), Some(b))),
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
/// which would keep integer/integer as integer. The row's "array-scalar" and
/// "transposed-vector–scalar" forms divide each element and keep the shape.
fn divide_type(a: Option<&Type>, b: Option<&Type>) -> Type {
    use ScalarType::*;
    match (a, b) {
        (Some(Type::Array { shape, elem }), Some(s @ Type::Scalar(_)))
            if matches!(**elem, Type::Scalar(_)) =>
        {
            return Type::Array {
                shape: shape.clone(),
                elem: Box::new(divide_type(Some(elem), Some(s))),
            };
        }
        (Some(Type::TVector { len, elem }), Some(s @ Type::Scalar(_)))
            if matches!(**elem, Type::Scalar(_)) =>
        {
            return Type::TVector {
                len: *len,
                elem: Box::new(divide_type(Some(elem), Some(s))),
            };
        }
        _ => {}
    }
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
    // An unresolved operand (`%any`, e.g. a lambda's placeholder) may itself be an array,
    // so only a KNOWN scalar scales an array: `L * x` with `x` unknown is not `L`'s type.
    let scalar = |t: &Type| matches!(t, Type::Scalar(_));
    match (a, b) {
        _ if scalarish(a) && scalarish(b) => promote2(Some(a), Some(b)),
        (Type::Array { .. }, s) if scalar(s) => a.clone(),
        (s, Type::Array { .. }) if scalar(s) => b.clone(),
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
/// rows do not admit under any calling convention.
fn is_array_like(t: &Type) -> bool {
    matches!(t, Type::Array { .. } | Type::TVector { .. })
}

/// The first operand of a comparison or a logical connective whose shape §07's
/// scalar domains do not admit, if any.
///
/// An array or a transposed vector never has a scalar reading. A `Table` counts only
/// when the call supplies more than the one positional argument §04 "Calling
/// conventions" splats — "Auto-splatting is shallow and occurs only when a record or
/// table is the call's sole argument" — so `gt(t, 1.0)` compares the table itself and
/// is refused, while `gt(table(a = x, b = y))` splats to `gt(a = x, b = y)` and its
/// operands are the columns, not the table.
///
/// A `Record` operand is still accepted, and the splat that carries ARRAY columns
/// into a scalar row (`gt(table(a = v1, b = v2))`) is still silent. Both are the same
/// class as this refusal and neither is fixed here — recorded in
/// `flatppl-dev/TODO-flatppl-rust.md` rather than argued away.
fn nonscalar_operand(args: &[ArgInfo], named: &[NamedInfo]) -> Option<Type> {
    let splatting = named.is_empty() && args.len() == 1;
    args.iter()
        .map(|(_, t, _)| t)
        .chain(named.iter().map(|(_, _, t, _)| t))
        .find(|t| is_array_like(t) || (!splatting && matches!(t, Type::Table { .. })))
        .cloned()
}

/// A short phrase naming a [`nonscalar_operand`] shape, for
/// [`refuse_nonscalar_operand`]'s message. `Type` has no `Display`, and the rank is
/// the part of the shape that makes the refusal legible.
fn nonscalar_shape_phrase(t: &Type) -> String {
    match t {
        Type::Array { shape, .. } => format!("a rank-{} array", shape.len()),
        Type::TVector { .. } => "a transposed vector".to_string(),
        Type::Table { columns, .. } => format!("a {}-column table", columns.len()),
        _ => "a non-scalar".to_string(),
    }
}

/// A non-scalar operand to `equal`/`unequal`/`lt`/`le`/`gt`/`ge` or to
/// `land`/`lor`/`lxor`/`lnot`.
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
/// §07 "Logic and conditionals" gives `land`, `lor`, `lxor` and `lnot` the domain
/// `booleans` — the same kind of scalar value-set, in a table whose only other row
/// (`ifelse`) spells its `anything` domain out. §05's broadcasting bullet does not
/// enumerate them, so the Domains column carries this refusal on its own.
///
/// This was accepted before, and the leniency was not harmless: `infer` typed
/// `gt(v, 3.0)` over a length-3 `v` as `Scalar(Boolean)` — a scalar — while the
/// StableHLO emitter broadcast the same node and emitted
/// `func.func @logdensity(%arg0: tensor<3xf32>) -> tensor<3xi1>`. The declared type
/// and the emitted ABI disagreed on the result's SHAPE, with no diagnostic on either
/// side. The connectives reached the same disagreement one step later:
/// `ifelse(lnot(gt.(v, 3.0)), 1.0, 2.0)` typed `Scalar(Real)` over an operand typed a
/// `[3]` boolean array, and the whole log-density compiled rank-lifted to
/// `tensor<3xf32>`, exit 0. The emitter's predicate gate cannot catch that — the four
/// connectives are `PREDICATE_HEADS` entries by design. Refusing statically closes
/// every half at once.
///
/// The diagnostic names the dotted form, which is the working route: `gt.(v, 3.0)`,
/// `v .> 3.0` and `broadcast(gt, v, 3.0)` all reach [`broadcast_type`]'s elementwise
/// arm and type a boolean array of `v`'s shape. A TABLE operand has no dotted route,
/// so its message names §03's column access instead.
fn refuse_nonscalar_operand(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    let offender = nonscalar_operand(args, named).unwrap_or(Type::Deferred);
    let citation = if matches!(name, "land" | "lor" | "lxor" | "lnot") {
        "spec §07 \"Logic and conditionals\" gives the logical operators the scalar \
         domain `booleans`"
            .to_string()
    } else {
        "spec §07 \"Operator-equivalent functions\" gives the comparisons the scalar \
         domains `reals` and `integers`/`booleans`/strings, where `add`/`sub` in the \
         same table read \"scalars or arrays of same shape\", and §05 \"Excluded \
         constructs\" states \"No implicit operator broadcasting\""
            .to_string()
    };
    let remedy = if matches!(offender, Type::Table { .. }) {
        "Select a column and compare it elementwise instead — §03 \"Indexing\" gives \
         column access as `t.colname` — which gives a boolean array of the column's \
         length"
            .to_string()
    } else {
        let dotted = if name == "lnot" {
            format!("`{name}.(a)`")
        } else {
            format!("`{name}.(a, b)`")
        };
        format!(
            "Apply it elementwise instead — {dotted}, or the dotted operator — which \
             gives a boolean array of the operand's shape"
        )
    };
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!(
            "`{name}` expects scalar operands, got {}: {citation}. {remedy}",
            nonscalar_shape_phrase(&offender)
        ),
    ));
    Type::Failed(format!("`{name}` applied to a non-scalar operand").into())
}

/// Every built-in head whose §07 Domains cell is a COLLECTION — arrays, vectors, matrices
/// or tables — and never a scalar, paired with the §07 table it lives in and that cell as
/// §07 writes it (maths rendered as text; a multi-argument row carries the whole cell even
/// though only the FIRST argument's domain drives the refusal). The table drives both
/// halves of the broadcast rule in [`broadcast_type`]: a scalar cell is refused against
/// the quoted domain, and an array cell gets the per-cell answer from
/// [`broadcast_collection_cell`].
///
/// A head is here because §07 denies it a scalar, not because it reduces. §07
/// "Cumulative operations" calls its four a scan that "preserve[s] the shape of their
/// input rather than reducing it", and the normalizations in §07 "Norms and
/// normalization" likewise return a vector — but all of them are spelled over a
/// vector, so a scalar cell misses their domain exactly as it misses `sum`'s. §07 "Linear
/// algebra" and §07 "Array and table operations" are here for the same reason and were
/// added a round later: `transpose.(v)`, `adjoint.(v)` and `self_outer.(v)` were MEASURED
/// discarding the wrapper and emitting the undotted head's answer at exit 0, the last of
/// them a `tensor<4x4xf32>` outer product.
///
/// Two absences are deliberate, both because the row admits a scalar:
///
/// - `min`/`max` — §07 "Operator-equivalent functions" gives that pair two scalar
///   arguments, so `max.(v, 0.0)` is a genuine elementwise broadcast and belongs to the
///   cell table below. The reductions are `maximum`/`minimum`.
/// - `cat` — §07 "Array and table operations" gives it "scalars, vectors, or records",
///   and "`cat(scalar1, scalar2, ...)` … produces a vector of those scalars". So
///   `cat.(v, w)` is a well-formed per-element `cat` of two scalars, not a domain error.
///
/// The nine tables swept are Reductions, Boolean reductions, Cumulative operations,
/// Norms and normalization, Linear algebra, Array and table operations, Approximation
/// functions, Binning, and Membership, filtering, and bin selection.
///
/// Convolution and array generation also have collection-domain arguments. Their
/// inclusion here keeps scalar-cell broadcasts from discarding that domain.
pub(crate) const COLLECTION_DOMAIN_HEADS: &[(&str, &str, &str)] = &[
    ("conv", "Convolution", "vector, vector"),
    ("crosscorr", "Convolution", "vector, vector"),
    ("sum", "Reductions", "real/complex arrays"),
    ("mean", "Reductions", "real/complex arrays"),
    ("var", "Reductions", "real arrays"),
    ("std", "Reductions", "real arrays"),
    ("prod", "Reductions", "real/complex arrays"),
    ("maximum", "Reductions", "real arrays"),
    ("minimum", "Reductions", "real arrays"),
    ("median", "Reductions", "real arrays"),
    ("quantile", "Reductions", "real arrays, `interval(0, 1)`"),
    ("lengthof", "Reductions", "vectors, tables"),
    (
        "array",
        "Array and table generation",
        "vector, integer vector, integer vector",
    ),
    ("sizeof", "Reductions", "vectors, arrays"),
    ("indicesof", "Reductions", "vectors, arrays, tables"),
    ("indicesof0", "Reductions", "vectors, arrays, tables"),
    ("lany", "Boolean reductions", "boolean arrays"),
    ("lall", "Boolean reductions", "boolean arrays"),
    ("cumsum", "Cumulative operations", "vectors"),
    ("cumprod", "Cumulative operations", "vectors"),
    ("cummax", "Cumulative operations", "real vectors"),
    ("cummin", "Cumulative operations", "real vectors"),
    ("l1norm", "Norms and normalization", "real/complex vectors"),
    ("l2norm", "Norms and normalization", "real/complex vectors"),
    (
        "linfnorm",
        "Norms and normalization",
        "real/complex vectors",
    ),
    ("l1unit", "Norms and normalization", "real/complex vectors"),
    ("l2unit", "Norms and normalization", "real/complex vectors"),
    ("logsumexp", "Norms and normalization", "real vectors"),
    ("softmax", "Norms and normalization", "real vectors"),
    ("logsoftmax", "Norms and normalization", "real vectors"),
    ("transpose", "Linear algebra", "vectors, matrices"),
    ("adjoint", "Linear algebra", "vectors, matrices"),
    ("det", "Linear algebra", "square matrices"),
    ("logabsdet", "Linear algebra", "square matrices"),
    ("inv", "Linear algebra", "square matrices"),
    ("trace", "Linear algebra", "square matrices"),
    ("linsolve", "Linear algebra", "square `A`, vector `b`"),
    ("qr", "Linear algebra", "m x n, m >= n matrices"),
    ("lower_cholesky", "Linear algebra", "positive definite `A`"),
    ("row_gram", "Linear algebra", "matrices"),
    ("col_gram", "Linear algebra", "matrices"),
    ("self_outer", "Linear algebra", "vectors"),
    (
        "cross",
        "Linear algebra",
        "real or complex vectors with `lengthof(a) == lengthof(b) == 3`",
    ),
    ("diagmat", "Linear algebra", "vectors"),
    ("diag", "Linear algebra", "matrices, integer"),
    ("quadform", "Linear algebra", "square `A`, vector `x`"),
    (
        "rowstack",
        "Array and table operations",
        "vector of equal-length vectors",
    ),
    (
        "colstack",
        "Array and table operations",
        "vector of equal-length vectors",
    ),
    (
        "tile",
        "Array and table operations",
        "array, integer or integer vector",
    ),
    (
        "splitblocks",
        "Array and table operations",
        "array, integer or integer vector",
    ),
    (
        "joinblocks",
        "Array and table operations",
        "array of equal-shaped arrays",
    ),
    (
        "partition",
        "Array and table operations",
        "vector, positive integer or integer vector",
    ),
    ("reverse", "Array and table operations", "vectors, tables"),
    (
        "addaxes",
        "Array and table operations",
        "array, non-negative integer, non-negative integer",
    ),
    (
        "blockdiagmat",
        "Array and table operations",
        "vector of matrices",
    ),
    (
        "bandedmat",
        "Array and table operations",
        "vector, positive integer",
    ),
    (
        "selectbins",
        "Membership, filtering, and bin selection",
        "vector, set, vector",
    ),
    (
        "bincounts",
        "Binning",
        "vector or record of edge vectors, array or record",
    ),
    (
        "polynomial",
        "Approximation functions",
        "vector, real or complex",
    ),
    (
        "bernstein",
        "Approximation functions",
        "vector, `unitinterval`",
    ),
    (
        "stepwise",
        "Approximation functions",
        "vector, vector, real",
    ),
];

pub(crate) fn collection_domain_head(name: &str) -> Option<(&'static str, &'static str)> {
    COLLECTION_DOMAIN_HEADS
        .iter()
        .find(|(h, _, _)| *h == name)
        .map(|(_, section, domain)| (*section, *domain))
}

/// `broadcast(<collection-domain head>, …)` whose cells are SCALARS — the whole
/// dotted-reduction family, refused against the §07 domain the cell misses.
///
/// §04 "Broadcasting" maps the head "elementwise over arrays", and its "Collection
/// arguments" paragraph iterates every array argument along every axis; only the
/// "Non-collection inputs" — scalars, functions, kernels, measures — are "held
/// constant". So the head sees an ELEMENT, and the elements of a real array are
/// scalars, which no row of [`COLLECTION_DOMAIN_HEADS`] admits.
///
/// This closes a silent mislowering, not a cosmetic gap. Every head in the table typed
/// `%deferred` with no diagnostic while `stablehlo::Emitter::lower_broadcast` handed
/// the head straight to `lower_builtin` — discarding the `broadcast` wrapper — so
/// `sum.(v)` compiled to the undotted `sum(v)`'s reduce and answered with a NUMBER at
/// exit 0, and `sum.(M)` contracted a matrix all the way to a scalar. The spellings the
/// message offers are what the writer could have meant, and §03's nested array is the
/// last of them: that one is LEGAL and typed by [`broadcast_collection_cell`].
fn refuse_broadcast_collection_cell(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    section: &str,
    domain: &str,
    cell: &Type,
) -> Type {
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!(
            "`{name}` under a broadcast is applied to each ELEMENT, and the elements here \
             are {}: spec §07 \"{section}\" gives `{name}` the domain \"{domain}\", and spec \
             §04 \"Broadcasting\" maps the head \"elementwise over arrays\". {} The dotted \
             spelling means something only over a NESTED array (§03 \"Arrays\"), whose \
             elements are themselves arrays.",
            scalar_cell_phrase(cell),
            collection_domain_remedy(name)
        ),
    ));
    Type::Failed(format!("`{name}` broadcast over scalar elements").into())
}

/// The ten built-ins §04 "Multi-axis aggregation" admits as an `aggregate` reduction:
/// "The eligible built-ins are `sum`, `prod`, `mean`, `var`, `std`, `maximum`,
/// `minimum`, `median`, `lany` and `lall`." Every other [`COLLECTION_DOMAIN_HEADS`]
/// entry is INELIGIBLE — §07 says as much for the cumulative four ("they are not
/// eligible reductions for multi-axis aggregation") — so the refusal above must not
/// offer them that remedy.
pub(crate) const AGGREGATE_ELIGIBLE_REDUCTIONS: &[&str] = &[
    "sum", "prod", "mean", "var", "std", "maximum", "minimum", "median", "lany", "lall",
];

/// The remedy sentence of a collection-domain refusal: the bare call to write instead,
/// with the head's §07 argument list, plus the per-axis `aggregate` form for the ten
/// heads §04 admits. `quantile` is §07's one two-argument row here, so `quantile(v)`
/// would be an arity error; and offering `aggregate(l2norm, …)` would send the reader
/// into §04's eligibility refusal instead of onto a working spelling.
///
/// The StableHLO emitter's `ops::collection_domain_remedy` is the same sentence for the
/// same reason, so the two refusals name the same remedies.
fn collection_domain_remedy(name: &str) -> String {
    let bare = if name == "quantile" {
        "quantile(v, p)".to_string()
    } else {
        format!("{name}(v)")
    };
    if AGGREGATE_ELIGIBLE_REDUCTIONS.contains(&name) {
        format!(
            "Use the bare `{bare}`, or `aggregate({name}, [.i], M[.i, .j])` \
             (§04 \"Multi-axis aggregation\") to reduce along one axis of a multi-axis array."
        )
    } else {
        format!("Use the bare `{bare}`.")
    }
}

fn scalar_cell_phrase(cell: &Type) -> String {
    match cell {
        Type::Scalar(s) => format!("{} scalars", scalar_word(*s)),
        _ => "scalars".to_string(),
    }
}

fn scalar_word(s: ScalarType) -> &'static str {
    match s {
        ScalarType::Real => "real",
        ScalarType::Integer => "integer",
        ScalarType::Boolean => "boolean",
        ScalarType::Complex => "complex",
    }
}

/// The per-cell result of a [`COLLECTION_DOMAIN_HEADS`] head over a cell that IS a
/// collection — the legal `sum.(vv)` over §03's vector of vectors, which reduces each
/// inner vector. `None` leaves the broadcast `%deferred`.
///
/// Scalar reductions and the supported collection results are answered here. `sum`/`prod`/`mean`/
/// `maximum`/`minimum` route through [`reduce_type`], so the dotted form and the bare
/// form cannot drift on §03's promotions — `sum` over boolean cells is `Integer` on
/// both. The rest read their §07 row directly: `var`/`std`/`median`/`quantile` and the
/// three norms are real-valued, `lany`/`lall` are truth values per §07 "Boolean
/// reductions", `lengthof` counts.
///
/// Heads without a cell rule, including multi-axis index tuples, remain deferred.
fn broadcast_collection_cell(name: &str, cell: &Type) -> Option<Type> {
    let elem = match cell {
        Type::Array { elem, .. } | Type::TVector { elem, .. } => elem.as_ref(),
        _ => return None,
    };
    match name {
        "sum" | "prod" | "mean" | "maximum" | "minimum" => match reduce_type(name, Some(cell)) {
            Type::Deferred => None,
            t => Some(t),
        },
        "var" | "std" | "median" | "quantile" | "l1norm" | "l2norm" | "linfnorm" | "logsumexp" => {
            matches!(elem, Type::Scalar(_)).then_some(Type::Scalar(ScalarType::Real))
        }
        "lany" | "lall" => Some(Type::Scalar(ScalarType::Boolean)),
        "lengthof" => Some(Type::Scalar(ScalarType::Integer)),
        "sizeof" => Some(sizeof_type(Some(cell))),
        "indicesof" | "indicesof0" => match indices_type(Some(cell)) {
            result @ Type::Array { .. } => Some(result),
            _ => None,
        },
        "det" | "logabsdet" => (flatten_dims(cell).len() == 2).then(|| {
            Type::Scalar(if name == "logabsdet" {
                ScalarType::Real
            } else {
                elem_scalar_kind_of(cell).unwrap_or(ScalarType::Real)
            })
        }),
        "lower_cholesky" | "inv" => {
            let shape = flatten_dims(cell);
            (shape.len() == 2).then(|| Type::Array {
                shape: shape.into_boxed_slice(),
                elem: Box::new(Type::Scalar(
                    elem_scalar_kind_of(cell).unwrap_or(ScalarType::Real),
                )),
            })
        }
        _ => None,
    }
}

fn sizeof_type(array: Option<&Type>) -> Type {
    let rank = match array {
        Some(Type::Array { shape, .. }) => shape.len(),
        Some(Type::TVector { .. }) => 1,
        _ => return Type::Deferred,
    };
    Type::Array {
        shape: Box::new([Dim::Static(rank as u32)]),
        elem: Box::new(Type::Scalar(ScalarType::Integer)),
    }
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

/// An array of equal-length vectors becomes the rows or columns of a matrix.
fn stack_type(a: Option<&Type>, columns: bool) -> Type {
    match a {
        Some(Type::Array { shape, elem }) if shape.len() == 1 => match elem.as_ref() {
            Type::Array {
                shape: inner,
                elem: cell,
            } if inner.len() == 1 => Type::Array {
                shape: Box::new(if columns {
                    [inner[0], shape[0]]
                } else {
                    [shape[0], inner[0]]
                }),
                elem: cell.clone(),
            },
            _ => Type::Deferred,
        },
        _ => Type::Deferred,
    }
}

fn convolution_type(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    args: &[ArgInfo],
) -> Type {
    let (Some(Type::Array { shape: v, elem: ve }), Some(Type::Array { shape: k, elem: ke })) =
        (arg_ty(args, 0), arg_ty(args, 1))
    else {
        return Type::Deferred;
    };
    if v.len() != 1 || k.len() != 1 {
        return Type::Deferred;
    }
    let length = match (v[0], k[0]) {
        (Dim::Static(n), Dim::Static(m)) if m > n => {
            inf.diags.push(crate::Diagnostic::error_at(
                id,
                format!("`{name}` requires kernel length at most signal length (spec §07)"),
            ));
            return Type::Failed(format!("{name}: kernel too long").into());
        }
        (Dim::Static(n), Dim::Static(m)) => n
            .checked_sub(m)
            .and_then(|x| x.checked_add(1))
            .map_or(Dim::Dynamic, Dim::Static),
        _ => Dim::Dynamic,
    };
    Type::Array {
        shape: Box::new([length]),
        elem: Box::new(promote2(Some(ve), Some(ke))),
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
/// So the entry type is whatever that reduction gives. [`reduced_scalar`]
/// supplies the same rule used by ordinary array and table reductions. Axis
/// indexing leaves the body node deferred, so [`aggregate_body_scalar_kind`]
/// traces through `get` and local references to recover its element kind.
///
/// - `median` — real even over an integer body, for [`reduced_scalar`]'s reason.
/// - `lany`/`lall` — boolean whatever the body was; §07 "Boolean reductions" gives
///   both a truth value.
///
/// Spec §04 "Metric-aware Einstein summation" / "Static checks": "bare neutral
/// aggregate axes (`.i` without a variance marker) are not allowed inside
/// `metricsum`." The same section states variance-marked axis names are
/// "required inside `metricsum`".
///
/// A neutral axis has no lowering here: "each `_` (lower-variance) axis name in
/// `expr` becomes an `inv(metric)` contraction" and an `^` axis reads
/// all-contravariant storage directly, so a bare `.i` names neither and the
/// contraction-pairing rule above it cannot be applied to it either. Absorbed
/// silently, it reached the determiniser as a plain `aggregate` axis, dropping
/// the metric factor the author's index placement asked for.
///
/// Both the `output_axes` list and the body are checked. The body walk stops at
/// a nested `aggregate` / `metricsum`, since the section scopes variance-marked
/// axis names "lexically ... to the enclosing `metricsum`" — a nested
/// `aggregate` keeps its own neutral axes, and a nested `metricsum` gets its own
/// visit here.
///
/// `Some` carries the `Failed` type for the call; `None` means every axis in
/// scope carries a marker.
fn metricsum_neutral_axis_check(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let mut bare = None;
    if let Some((axes, _, _)) = args.get(1) {
        neutral_axis_scan(inf, *axes, &mut bare);
    }
    if bare.is_none()
        && let Some((body, _, _)) = args.get(2)
    {
        neutral_axis_scan(inf, *body, &mut bare);
    }
    let (node, name) = bare?;
    inf.diags.push(crate::Diagnostic::error_at(
        node,
        format!(
            "bare neutral axis `.{name}` is not allowed inside `metricsum` (spec §04 \
             \"Metric-aware Einstein summation\", \"Static checks\": \"bare neutral \
             aggregate axes (`.i` without a variance marker) are not allowed inside \
             `metricsum`\"); mark it `.{name}^` for an upper (contravariant) index or \
             `.{name}_` for a lower (covariant) one"
        ),
    ));
    Some(Type::Failed("neutral axis inside metricsum".into()))
}

/// Walk for the first unmarked axis name in a `metricsum`'s own scope. First
/// offender wins: a whole body of neutral axes is one mistake, and naming them
/// all would bury the fix.
fn neutral_axis_scan(inf: &Inferencer<'_, '_>, node: NodeId, bare: &mut Option<(NodeId, String)>) {
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if bare.is_some() {
            return;
        }
        if inf.is_substituted(node) {
            continue;
        }
        if let Node::Axis(ax) = inf.module.node(node) {
            if ax.variance.is_none() {
                *bare = Some((node, inf.module.resolve(ax.name).to_string()));
            }
            continue;
        }
        let Node::Call(c) = inf.module.node(node) else {
            continue;
        };
        if let flatppl_core::CallHead::Builtin(op) = c.head {
            let name = inf.module.resolve(op);
            if name == "aggregate" || name == "metricsum" {
                continue;
            }
        }
        pending.extend(c.named.iter().rev().map(|n| n.value));
        pending.extend(c.args.iter().rev().copied());
    }
}

/// `metricsum`'s metric must be a rank-2 array of scalars — spec §04
/// "Metric-aware Einstein summation": "It must be a square, symmetric, and
/// invertible rank-2 array", and, under "Expression restrictions", "`metric`
/// itself and all arrays indexed with co-/contravariant axis names in `expr`
/// must be arrays of scalars."
///
/// The nested-literal case is the one worth its own message. §03 "Arrays":
/// "Vectors of vectors are not interpreted as matrices implicitly, but can be
/// turned into matrices explicitly using `rowstack` or `colstack`." So
/// `[[1.0, 0.0], [0.0, -1.0]]` is a vector of two vectors, and the two lifts
/// disagree on which axis the inner vectors become — the user has to pick, so
/// this pass cannot pick for them.
///
/// `axis_count` is deliberately NOT used here: it flattens a nested array's
/// axes into the count, so it reads a vector of vectors as rank 2 and would
/// wave through exactly the type this check exists to reject.
///
/// `Some` carries the `Failed` type for the call; `None` means the metric is
/// either well-typed or not yet known.
fn metricsum_metric_check(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let (node, ty, _) = args.first()?;
    let (complaint, reason) = match ty {
        Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_) => return None,
        Type::Array { shape, elem } if shape.len() == 2 => match elem.as_ref() {
            Type::Scalar(_) | Type::Deferred | Type::Any | Type::Var(_) => return None,
            _ => (
                metric_rank_complaint(ty),
                "metricsum metric is not a rank-2 array of scalars",
            ),
        },
        Type::Array { shape, elem } if shape.len() == 1 && matches!(**elem, Type::Array { .. }) => {
            (
                "`metricsum`'s metric is a vector of vectors, which spec §03 \"Arrays\" \
             does not read as a matrix: \"Vectors of vectors are not interpreted as \
             matrices implicitly, but can be turned into matrices explicitly using \
             `rowstack` or `colstack`\". Wrap it — `rowstack(...)` makes the inner \
             vectors the ROWS, `colstack(...)` the COLUMNS — since the two disagree \
             and only you know which storage order you meant"
                    .to_string(),
                "metricsum metric is a vector of vectors",
            )
        }
        other => (
            metric_rank_complaint(other),
            "metricsum metric is not a rank-2 array of scalars",
        ),
    };
    inf.diags
        .push(crate::Diagnostic::error_at(*node, complaint));
    Some(Type::Failed(reason.into()))
}

/// The generic half of [`metricsum_metric_check`]'s refusal, for every metric
/// that is not a vector of vectors — that case gets its own message.
fn metric_rank_complaint(ty: &Type) -> String {
    let got = match ty {
        Type::Scalar(s) => format!("a {} scalar", scalar_word(*s)),
        Type::Array { shape, elem } if matches!(**elem, Type::Array { .. }) => {
            format!("a rank-{} array of arrays", shape.len())
        }
        Type::Array { shape, .. } => format!("a rank-{} array", shape.len()),
        other => nonscalar_shape_phrase(other),
    };
    format!(
        "`metricsum`'s metric must be a rank-2 array of scalars (spec §04 \
         \"Metric-aware Einstein summation\": \"It must be a square, symmetric, and \
         invertible rank-2 array\"); got {got}"
    )
}

/// The `expr` half of the same sentence [`metricsum_metric_check`] enforces —
/// spec §04 "Metric-aware Einstein summation", "Expression restrictions":
/// "`metric` itself and all arrays indexed with co-/contravariant axis names in
/// `expr` must be arrays of scalars."
///
/// The sentence names co-/contravariant axis names, so a container reached only
/// through neutral `aggregate` axes is outside it. Variance-marked axis names
/// are "lexically scoped to the enclosing `metricsum`", so the walk stops at a
/// nested `aggregate` / `metricsum` — that call gets its own visit here.
///
/// `axis_count` and `flatten_dims` are deliberately NOT used: both flatten a
/// nested array's axes into the count, so they read a vector of vectors as
/// rank 2 and would wave through exactly the type this check exists to reject.
///
/// `Some` carries the `Failed` type for the call; `None` means every
/// variance-indexed container is either well-typed or not yet known.
fn metricsum_expr_check(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let mut bad = None;
    variance_container_scan(inf, args.get(2)?.0, &mut bad);
    let (node, complaint, reason) = bad?;
    inf.diags.push(crate::Diagnostic::error_at(node, complaint));
    Some(Type::Failed(reason.into()))
}

/// Walk a `metricsum` body for the first container indexed by a variance-marked
/// axis that is not an array of scalars. First offender wins — the rest of the
/// body would cascade off the same nesting mistake.
fn variance_container_scan(
    inf: &mut Inferencer<'_, '_>,
    node: NodeId,
    bad: &mut Option<(NodeId, String, &'static str)>,
) {
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if bad.is_some() {
            return;
        }
        if inf.is_substituted(node) {
            continue;
        }
        let Node::Call(c) = inf.module.node(node) else {
            continue;
        };
        if let flatppl_core::CallHead::Builtin(op) = c.head {
            let name = inf.module.resolve(op);
            if name == "aggregate" || name == "metricsum" {
                continue;
            }
            if (name == "get" || name == "get0") && c.args.len() >= 2 {
                let variance_marked = c.args[1..].iter().any(
                    |&i| matches!(inf.module.node(i), Node::Axis(ax) if ax.variance.is_some()),
                );
                if variance_marked {
                    *bad = variance_container_complaint(inf, c.args[0]);
                }
            }
        }
        pending.extend(c.named.iter().rev().map(|n| n.value));
        pending.extend(c.args.iter().rev().copied());
    }
}

/// The refusal for one variance-indexed container, or `None` when it is an array
/// of scalars or its type is not yet known. A container that is not an array at
/// all is a different error, left to `get`'s own rule.
fn variance_container_complaint(
    inf: &Inferencer<'_, '_>,
    container: NodeId,
) -> Option<(NodeId, String, &'static str)> {
    let ty = inf.node_annotation(container).0;
    let Type::Array { elem, .. } = &ty else {
        return None;
    };
    match elem.as_ref() {
        Type::Scalar(_) | Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_) => None,
        Type::Array { .. } => Some((
            container,
            "this `metricsum` tensor is a vector of vectors, which spec §03 \"Arrays\" \
             does not read as a matrix: \"Vectors of vectors are not interpreted as \
             matrices implicitly, but can be turned into matrices explicitly using \
             `rowstack` or `colstack`\". Spec §04 \"Expression restrictions\" requires \
             every array indexed with co-/contravariant axis names to be an array of \
             scalars, so wrap it — `rowstack(...)` makes the inner vectors the ROWS, \
             `colstack(...)` the COLUMNS — since the two disagree and only you know \
             which storage order you meant"
                .to_string(),
            "metricsum tensor is a vector of vectors",
        )),
        other => Some((
            container,
            format!(
                "an array indexed with co-/contravariant axis names must be an array of \
                 scalars (spec §04 \"Expression restrictions\": \"`metric` itself and all \
                 arrays indexed with co-/contravariant axis names in `expr` must be arrays \
                 of scalars\"); got an array whose element is {}",
                nonscalar_shape_phrase(other)
            ),
            "metricsum tensor is not an array of scalars",
        )),
    }
}

/// Spec §04 "Metric-aware Einstein summation" / "Static checks": "every output
/// index must occur in `expr` with the same variance and may not also be
/// contracted".
///
/// The two halves fail differently, so they get separate messages. An output
/// index absent from `expr` names a component the body never computes. An
/// output index that also occurs with the opposite marker asks for a
/// contraction over an index the result still carries — under "Lowering to
/// `aggregate`" a lower output axis becomes a `metric` contraction *after* the
/// sum, so the same name cannot be both summed away and raised.
///
/// `Some` carries the `Failed` type for the call; `None` means every output
/// index occurs as declared.
fn metricsum_output_index_check(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let declared = metricsum_output_axes(inf, args.get(1)?.0)?;
    let mut occurrences = Vec::new();
    variance_axis_occurrences(inf, args.get(2)?.0, &mut occurrences);
    for &(decl_node, name, variance) in declared.iter() {
        let mut matched = false;
        let mut opposite = None;
        for &(node, occ_name, occ_variance) in occurrences.iter() {
            if occ_name != name {
                continue;
            }
            if occ_variance == variance {
                matched = true;
            } else if opposite.is_none() {
                opposite = Some(node);
            }
        }
        let spelling = axis_spelling(inf, name, variance);
        let (node, complaint, reason) = if let Some(node) = opposite {
            (
                node,
                format!(
                    "output axis `{spelling}` also occurs in `expr` as \
                     `{}`, which contracts an index the result still carries \
                     (spec §04 \"Metric-aware Einstein summation\", \"Static checks\": \
                     \"every output index must occur in `expr` with the same variance and \
                     may not also be contracted\"); either match the output variance here \
                     or rename this occurrence to a contracted index",
                    axis_spelling(inf, name, flip_variance(variance))
                ),
                "metricsum output index is also contracted",
            )
        } else if matched {
            continue;
        } else {
            (
                decl_node,
                format!(
                    "output axis `{spelling}` does not occur in `expr` (spec §04 \
                     \"Metric-aware Einstein summation\", \"Static checks\": \"every output \
                     index must occur in `expr` with the same variance\"); `expr` has to \
                     compute the components the output axes name"
                ),
                "metricsum output index is absent from expr",
            )
        };
        inf.diags.push(crate::Diagnostic::error_at(node, complaint));
        return Some(Type::Failed(reason.into()));
    }
    None
}

/// Spec §04 "Metric-aware Einstein summation" / "Static checks": "Every
/// repeated non-output index in `expr` must occur exactly twice — once upper
/// and once lower".
///
/// The clause governs *repeated* indices only, so a non-output index occurring
/// ONCE is left alone: "Equivalence to `aggregate` under identity metric" makes
/// `metricsum(eye(n), ...)` an `aggregate(sum, ...)`, and a plain row sum
/// (`aggregate(sum, [.mu], A[.mu, .nu])`) spells exactly that — one unpaired
/// non-output index. Refusing it would refuse a construct the equivalence
/// clause requires. `flatppl-js` `analyzer.ts` accepts it too, since #219
/// relaxed its static check #3.
///
/// A repeated index with no metric factor between its two ends is the real
/// error: under "Lowering to `aggregate`" only a `_` axis inserts
/// `inv(metric)`, so two uppers or two lowers sum stored components with no
/// metric at all, and a third occurrence leaves the pairing ambiguous.
///
/// `Some` carries the `Failed` type for the call; `None` means every repeated
/// non-output index is one upper and one lower.
fn metricsum_contraction_check(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let declared = metricsum_output_axes(inf, args.get(1)?.0)?;
    let mut occurrences = Vec::new();
    variance_axis_occurrences(inf, args.get(2)?.0, &mut occurrences);
    // First offender in source order: the rest of the body usually repeats the
    // same index mistake, and naming them all would bury the fix.
    for &(node, name, _) in occurrences.iter() {
        if declared.iter().any(|&(_, out_name, _)| out_name == name) {
            continue;
        }
        let uppers = occurrences
            .iter()
            .filter(|&&(_, n, v)| n == name && v == Variance::Upper)
            .count();
        let lowers = occurrences
            .iter()
            .filter(|&&(_, n, v)| n == name && v == Variance::Lower)
            .count();
        if uppers + lowers < 2 || (uppers == 1 && lowers == 1) {
            continue;
        }
        let bare = inf.module.resolve(name).to_string();
        inf.diags.push(crate::Diagnostic::error_at(
            node,
            format!(
                "contracted axis `.{bare}` must occur exactly twice in `expr` — once upper \
                 (`.{bare}^`) and once lower (`.{bare}_`) — but occurs {uppers} upper and \
                 {lowers} lower (spec §04 \"Metric-aware Einstein summation\", \"Static \
                 checks\": \"Every repeated non-output index in `expr` must occur exactly \
                 twice — once upper and once lower\"); only a `_` axis inserts the \
                 `inv(metric)` factor a contraction needs"
            ),
        ));
        return Some(Type::Failed("metricsum index is not paired".into()));
    }
    None
}

/// Spec §05 "Note on axis names": "The grammar likewise admits `AxisList` as a
/// `Primary`, but it is legal only as the `output_axes` argument of an
/// `aggregate` or `metricsum` call and as the axis-list binder of an
/// `AggregateBinding` or `MetricsumBinding`; anywhere else it is a static
/// error."
///
/// So `output_axes` has to be a literal axis list. Axis names are not values —
/// §05 "Axis names and aggregation": "an axis name is legal only as an entry in
/// `aggregate`'s `output_axes` axis list, as an index inside `[...]` within the
/// body, or as a binder on the left-hand side of `:=`" — so nothing else can
/// ever produce one, and `metricsum(g, axes, ...)` is a static error whatever
/// `axes` is bound to.
fn metricsum_output_axes_check(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let node = args.get(1)?.0;
    if metricsum_output_axes(inf, node).is_some() {
        return None;
    }
    inf.diags.push(crate::Diagnostic::error_at(
        node,
        "`metricsum`'s `output_axes` must be a literal axis list (spec §05 \"Note on axis \
         names\": \"The grammar likewise admits `AxisList` as a `Primary`, but it is legal \
         only as the `output_axes` argument of an `aggregate` or `metricsum` call and as the \
         axis-list binder of an `AggregateBinding` or `MetricsumBinding`; anywhere else it is \
         a static error\"); axis names are not values, so no binding or call can produce this \
         list — spell the axes here, `metricsum(metric, [.mu^, .nu_], expr)`"
            .to_string(),
    ));
    Some(Type::Failed(
        "metricsum output_axes is not a literal axis list".into(),
    ))
}

/// `output_axes` as `(node, name, variance)` for every variance-marked entry,
/// or `None` when arg 1 is not an axis-list literal — which
/// [`metricsum_output_axes_check`] has already refused, so the two index checks
/// below never run on one.
///
/// Unmarked entries are dropped: `metricsum_neutral_axis_check` already refused
/// the call, so this is only reached with markers everywhere.
fn metricsum_output_axes(
    inf: &Inferencer<'_, '_>,
    node: NodeId,
) -> Option<Vec<(NodeId, Symbol, Variance)>> {
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
        // An `ArrayLiteral` prints as the same `vector` call, so a non-`Axis`
        // entry means this is not an `AxisList` at all.
        let Node::Axis(ax) = inf.module.node(a) else {
            return None;
        };
        if let Some(v) = ax.variance {
            out.push((a, ax.name, v));
        }
    }
    Some(out)
}

/// Every variance-marked axis occurrence in a `metricsum` body, in source
/// order. Same scope boundary as [`neutral_axis_scan`]: §04 scopes
/// variance-marked axis names "lexically ... to the enclosing `metricsum`", so
/// the walk stops at a nested `aggregate` / `metricsum` and that call gets its
/// own visit.
fn variance_axis_occurrences(
    inf: &Inferencer<'_, '_>,
    node: NodeId,
    out: &mut Vec<(NodeId, Symbol, Variance)>,
) {
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if inf.is_substituted(node) {
            continue;
        }
        if let Node::Axis(ax) = inf.module.node(node) {
            if let Some(v) = ax.variance {
                out.push((node, ax.name, v));
            }
            continue;
        }
        let Node::Call(c) = inf.module.node(node) else {
            continue;
        };
        if let flatppl_core::CallHead::Builtin(op) = c.head {
            let name = inf.module.resolve(op);
            if name == "aggregate" || name == "metricsum" {
                continue;
            }
        }
        pending.extend(c.named.iter().rev().map(|n| n.value));
        pending.extend(c.args.iter().rev().copied());
    }
}

/// An axis as the source spells it — `.mu^` upper, `.mu_` lower.
fn axis_spelling(inf: &Inferencer<'_, '_>, name: Symbol, variance: Variance) -> String {
    let marker = match variance {
        Variance::Upper => '^',
        Variance::Lower => '_',
    };
    format!(".{}{marker}", inf.module.resolve(name))
}

fn flip_variance(v: Variance) -> Variance {
    match v {
        Variance::Upper => Variance::Lower,
        Variance::Lower => Variance::Upper,
    }
}

fn aggregate_result_kind(inf: &Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<ScalarType> {
    let Node::Const(op) = inf.module.node(args.first()?.0) else {
        return None;
    };
    let head = inf.module.resolve(*op);
    match head {
        "median" => Some(ScalarType::Real),
        "lany" | "lall" => Some(ScalarType::Boolean),
        "sum" | "prod" | "maximum" | "minimum" | "mean" | "var" | "std" => args
            .get(2)
            .and_then(|a| aggregate_body_scalar_kind(inf, a.0))
            .map(|elem| reduced_scalar(head, elem)),
        _ => None,
    }
}

/// Find the scalar kind beneath an aggregate body. Axis indexing leaves `get`
/// itself deferred, so follow only its collection operand and local binding
/// references. The fixed bound keeps malformed reference cycles harmless.
fn aggregate_body_scalar_kind(inf: &Inferencer<'_, '_>, mut node: NodeId) -> Option<ScalarType> {
    for _ in 0..64 {
        if let Some(kind) = inf.lookup_type(node).and_then(elem_scalar_kind_of) {
            return Some(kind);
        }
        node = match inf.module.node(node) {
            Node::Call(call) if matches!(call.head, CallHead::Builtin(op) if inf.module.resolve(op) == "get") => {
                *call.args.first()?
            }
            Node::Ref(r) if matches!(r.ns, flatppl_core::RefNs::SelfMod) => {
                let binding = inf.module.binding_by_name(r.name)?;
                inf.module.declared_binding_rhs(binding)
            }
            _ => return None,
        };
    }
    None
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

/// Record a static projection error at its selector node.
fn failed_get(inf: &mut Inferencer<'_, '_>, node: NodeId, message: String) -> Type {
    inf.diags
        .push(crate::Diagnostic::error_at(node, message.clone()));
    Type::Failed(message.into())
}

/// `get` with static selectors: integer indices consume array axes / pick
/// tuple components; string keys pick record fields. Anything dynamic or
/// sliced (`all` / `only` / axes) is deferred until the shape work.
fn get_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo], base: i64) -> Type {
    let Some((_, container, _)) = args.first() else {
        return Type::Deferred;
    };
    let mut current = container.clone();
    for (index, (node, _, _)) in args[1..].iter().enumerate() {
        if let Type::Array { shape, elem } = &current {
            return get_array_type(inf, shape, elem, &args[index + 1..], base);
        }
        let selector = inf.module.node(*node).clone();
        current = match (&current, &selector) {
            (Type::Tuple(comps), Node::Lit(Scalar::Int(k))) => {
                match usize::try_from(k - base).ok().and_then(|i| comps.get(i)) {
                    Some(t) => t.clone(),
                    None => return failed_get(inf, *node, "tuple index out of range".into()),
                }
            }
            (Type::TVector { elem, .. }, Node::Lit(Scalar::Int(_))) => elem.as_ref().clone(),
            (Type::Record(fields), Node::Lit(Scalar::Str(s))) => {
                let sym = fields.iter().find(|(n, _)| inf.module.resolve(*n) == &**s);
                match sym {
                    Some((_, t)) => t.clone(),
                    None => return failed_get(inf, *node, format!("record has no field `{s}`")),
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
            // A runtime integer row index selects one row in the same way.
            (Type::Table { columns, .. }, _)
                if matches!(args[index + 1].1, Type::Scalar(ScalarType::Integer)) =>
            {
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
                    None => return failed_get(inf, *node, format!("table has no column `{s}`")),
                }
            }
            (Type::Any | Type::Deferred, _) => return current.clone(),
            _ => return Type::Deferred,
        };
    }
    current
}

/// Each selector consumes one source axis. Gather and `all` retain result
/// axes; later selectors address subsequent source axes, not retained ones.
fn get_array_type(
    inf: &mut Inferencer<'_, '_>,
    shape: &[Dim],
    elem: &Type,
    selectors: &[ArgInfo],
    base: i64,
) -> Type {
    let mut retained = Vec::new();
    for (dim, (node, ty, _)) in shape.iter().zip(selectors) {
        match inf.module.node(*node) {
            Node::Const(name) if inf.module.resolve(*name) == "all" => retained.push(*dim),
            Node::Axis(_) | Node::Lit(Scalar::Int(_)) => {}
            _ => match ty {
                Type::Scalar(ScalarType::Integer) => {}
                Type::Array { shape, elem }
                    if matches!(elem.as_ref(), Type::Scalar(ScalarType::Integer)) =>
                {
                    retained.extend_from_slice(shape);
                }
                _ => return Type::Deferred,
            },
        }
    }
    let consumed = shape.len().min(selectors.len());
    retained.extend_from_slice(&shape[consumed..]);
    let elem = if selectors.len() > consumed {
        let mut args = vec![(selectors[consumed].0, elem.clone(), Phase::Fixed)];
        args.extend_from_slice(&selectors[consumed..]);
        get_type(inf, &args, base)
    } else {
        elem.clone()
    };
    if retained.is_empty() {
        elem
    } else {
        Type::Array {
            shape: retained.into(),
            elem: Box::new(elem),
        }
    }
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

fn indices_type(input: Option<&Type>) -> Type {
    let vector = |dim| Type::Array {
        shape: Box::new([dim]),
        elem: Box::new(Type::Scalar(ScalarType::Integer)),
    };
    match input {
        Some(Type::Array { shape, .. }) if shape.len() == 1 => vector(shape[0]),
        Some(Type::Array { shape, .. }) => Type::Tuple(shape.iter().copied().map(vector).collect()),
        Some(Type::TVector { len, .. } | Type::Table { nrows: len, .. }) => vector(*len),
        _ => Type::Deferred,
    }
}

/// The law of a value is normalized. Non-value objects are rejected separately.
fn lawof_type(arg: Option<&Type>) -> Type {
    Type::Measure {
        domain: Box::new(arg.cloned().unwrap_or(Type::Any)),
        mass: Mass::Normalized,
    }
}

/// A measure's non-probability mass class, for the gate diagnostic. Deferred
/// means not yet inferred, so it passes. Kernel admission belongs to lawof's
/// pointwise rule, not this shared measure gate.
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

/// Drawing requires a probability measure; normalization must be explicit.
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

/// Concrete random sampling still requires a probability measure.
fn rand_mass_gate(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let offending = unprovable_normalization(args)?;
    let (arg_node, _, _) = args.first()?;
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

/// `lawof` reifies a value node; measures and callables are already objects.
fn lawof_value_gate(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let (arg_node, arg_ty, _) = args.first()?;
    if !matches!(
        arg_ty,
        Type::Measure { .. }
            | Type::Kernel { .. }
            | Type::Function { .. }
            | Type::Likelihood { .. }
            | Type::Tuple(_)
            | Type::Module
    ) {
        return None;
    }
    inf.diags.push(crate::Diagnostic::error_at(
        *arg_node,
        "`lawof` requires a value node; use the measure directly, or take the law \
         of an explicit draw from a probability measure",
    ));
    Some(Type::Failed("lawof of a non-value object".into()))
}

/// Nodes one read of a set expression may resolve. A **work** budget, not a depth
/// bound: following a `%ref` lets a module share a subexpression, so `s_k =
/// cartprod(s_{k-1}, s_{k-1})` traverses $2^k$ nodes along paths only $k$ deep, and no
/// depth limit sees it coming (measured: 501 bytes at k=20 cost 5.0s and 847MB). The
/// budget is what the exponent runs out of. Memoizing the traversal would not be
/// enough on its own — the resulting `Type`/`ValueSet` is itself exponentially large,
/// so the size of the answer has to be bounded too, and every construction step here
/// spends budget.
///
/// Each entry point starts a fresh budget, so a given node reads the same way however
/// much work its siblings did. Exhaustion reads out as `%deferred`/`%unknown` — the
/// same honest "no answer" a set this reader cannot follow already gets.
///
/// This bounds WORK, which is not the same as bounding STACK — see
/// [`SET_EXPR_MAX_DEPTH`]. Both guards are needed.
const SET_EXPR_NODE_BUDGET: u32 = 10_000;

/// Reader frames one read of a set expression may nest. This bounds STACK, which the
/// work budget does not: a single chain spends one budget unit per hop but one stack
/// frame per hop too, so a budget alone lets `s = s` — or a long alias chain — recurse
/// until the stack dies, aborting the process (SIGABRT) before the cycle detector can
/// report `binding `s` is part of a reference cycle`. Measured with the budget as the
/// only guard: `s = s` aborted with exit 134, and a legal alias chain died at N≈8000 on
/// the main stack and between N=1500 (survived) and N=2000 (aborted) on a **2MB thread
/// stack** — the size `crates/lsp`, `crates/wasm-api` and libtest threads run on.
///
/// 256 leaves roughly a 6x margin under that 2MB limit while being 4x more nesting than
/// the deepest plausible set expression. Charged on every recursive step, so the
/// constant means exactly what it says: at most this many nested reader frames.
const SET_EXPR_MAX_DEPTH: u32 = 256;

/// The element type of a set expression (`elementof` / `external` argument),
/// read structurally — sets are not first-class in the type grammar.
pub(crate) fn set_element_type(inf: &mut Inferencer<'_, '_>, node: Option<NodeId>) -> Type {
    let mut budget = SET_EXPR_NODE_BUDGET;
    set_element_type_at(inf, node, &mut budget, 0)
}

fn set_element_type_at(
    inf: &mut Inferencer<'_, '_>,
    node: Option<NodeId>,
    budget: &mut u32,
    depth: u32,
) -> Type {
    let Some(node) = node else {
        return Type::Deferred;
    };
    // Work and stack are bounded separately; whichever runs out first defers.
    if *budget == 0 || depth > SET_EXPR_MAX_DEPTH {
        return Type::Deferred;
    }
    *budget -= 1;
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
                            let elem = set_element_type_at(inf, set_arg, budget, depth + 1);
                            // An unreadable element set makes the POWER unreadable, so
                            // defer whole rather than wrapping `%deferred` in a shape.
                            // `set_call_valueset`'s `cartpow` arm already bails this way
                            // on an `Unknown` element, and the two slots have to agree:
                            // a half-resolved `(%array 1 (4) %deferred)` beside a
                            // `%unknown` value set describes no set at all.
                            if matches!(elem, Type::Deferred) {
                                Type::Deferred
                            } else {
                                let shape = count_dims(inf, size_arg);
                                table_of_record_power(shape, elem)
                            }
                        }
                    }
                }
                // `ordered(n)` and `zerosum(n)` are subsets of ℝⁿ (§03 "Ordered
                // vectors", "Zero-sum vectors"): the element is a length-n real
                // vector and the constraint lives in the value-set slot.
                "stdsimplex" | "ordered" | "zerosum" => {
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
                            .map(|na| {
                                (
                                    na.name,
                                    set_element_type_at(inf, Some(na.value), budget, depth + 1),
                                )
                            })
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
                            .map(|&a| set_element_type_at(inf, Some(a), budget, depth + 1))
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
        // A set bound to a name — §03 "Preset domains" binds a set expression to
        // a global name, and §04 admits a name reference wherever the expression
        // itself is admitted, so `elementof(S_name)` must type as
        // `elementof(<S_name's rhs>)`.
        Node::Ref(r) if r.ns == RefNs::SelfMod => {
            let name = r.name;
            match module.binding_by_name(name) {
                Some(b) => {
                    let rhs = inf.module.binding(b).rhs;
                    set_element_type_at(inf, Some(rhs), budget, depth + 1)
                }
                None => Type::Deferred,
            }
        }
        Node::Ref(r) if matches!(r.ns, RefNs::Module(_)) => inf
            .module_set_element(node)
            .cloned()
            .unwrap_or(Type::Deferred),
        _ => Type::Deferred,
    }
}

/// The domain of a measure type, for `draw` / `rand`.
fn measure_domain(m: Option<&Type>) -> Type {
    match m {
        Some(Type::Measure { domain, .. }) => domain.as_ref().clone(),
        // A failed measure fails its variate too, rather than reading as "no rule yet".
        Some(failed @ Type::Failed(_)) => failed.clone(),
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
        (Some(Type::Measure { domain, .. }), None) => match domain.as_ref() {
            Type::Array { shape, elem } if shape.len() == 1 => {
                let selector = resolve_binding_refs(inf, call.args[0]);
                let selected = match inf.module.node(selector) {
                    Node::Lit(Scalar::Int(_)) => Some(1),
                    Node::Call(c)
                        if matches!(c.head, CallHead::Builtin(op) if inf.module.resolve(op) == "vector")
                            && c.args.iter().all(|n| {
                                matches!(inf.module.node(*n), Node::Lit(Scalar::Int(_)))
                            }) =>
                    {
                        Some(c.args.len())
                    }
                    _ => None,
                };
                let n = match (shape[0], selected) {
                    (Dim::Static(n), Some(k)) if n > k as u32 => Dim::Static(n - k as u32),
                    _ => Dim::Dynamic,
                };
                (
                    Box::new([inf.module.intern("complement")]),
                    Type::Array {
                        shape: Box::new([n]),
                        elem: elem.clone(),
                    },
                )
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
    let (shape, resolved_zero) = count_dims_with_zero(inf, *count_node);
    // spec §06 iid: a scalar size derived to 0 is an error over a
    // record-valued M — "a table has no zero-row form" — unlike the empty
    // product measure a scalar-valued M gets. §06 already requires a
    // positive WRITTEN size, so a zero here only ever comes from a DERIVED
    // size the const-eval table resolved (engine-concepts §17.1); a size
    // still open at analysis time stays `%dynamic` and is untouched. The
    // resolved 0 arrives as the flag, not as the dim: a derived 0 types
    // `%dynamic` per the empty-arrays ruling.
    if let (Type::Record(_), [_]) = (&domain, shape.as_ref())
        && resolved_zero
    {
        inf.diags.push(crate::Diagnostic::error_at(
            *count_node,
            "`iid`'s size resolves to 0 over a record-valued measure (spec \
             §06 iid): a table has no zero-row form"
                .to_string(),
        ));
        return Type::Failed("iid: derived size 0 over a record-valued measure".into());
    }
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

/// `PoissonProcess(intensity = M)` (spec §08): a measure over point
/// CONFIGURATIONS drawn from the intensity's own point space. §08 gives the
/// variate as "arrays (scalar points) or tables (record-valued points)", so the
/// domain is [`iid_type`]'s array-or-table rule applied to `M`'s domain — a
/// record point space gives a table (§03's Cartesian power over a record set
/// "is the set of tables with those columns"), anything else an array.
///
/// The length is `Dim::Dynamic`, and that is not a degrade: the event count is
/// itself random, so `PoissonProcess` carries no count argument for a static
/// shape to come from. §08 also makes the order carry no meaning
/// (permutation-invariant), which no `Type` variant expresses.
///
/// Mass is left `Deferred` for `fill_mass`'s §08 catch-all to make
/// `%normalized`, which is right: the mass over all configurations is
/// $\sum_k e^{-\Lambda} \Lambda^k / k! = 1$. A KERNEL intensity (§08 admits
/// one) carries no domain in `Type::Kernel`, so it stays `%deferred`.
fn poisson_process_type(
    inf: &mut Inferencer<'_, '_>,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    let intensity = named
        .iter()
        .find(|(n, _, _, _)| inf.module.resolve(*n) == "intensity")
        .map(|(_, _, t, _)| t.clone())
        .or_else(|| arg_ty(args, 0).cloned());
    let Some(Type::Measure { domain, .. }) = intensity else {
        return Type::Deferred;
    };
    let config = match *domain {
        Type::Record(fields) => Type::Table {
            columns: fields,
            nrows: Dim::Dynamic,
        },
        point => Type::Array {
            shape: Box::new([Dim::Dynamic]),
            elem: Box::new(point),
        },
    };
    Type::Measure {
        domain: Box::new(config),
        mass: Mass::Deferred,
    }
}

/// `BinnedPoissonProcess(bins, intensity)` (spec §08): "the pushforward of a
/// `PoissonProcess` through `bincounts`", so the variate is the count array §07
/// `bincounts(bins, _)` returns. The catalogue row's `BinCounts` domain builds it.
///
/// §08 gives `bins` "Same format as for `bincounts`", an edge vector or a record of
/// edge vectors, and `intensity` "finite-mass measure or kernel". Anything else is
/// refused. A kernel intensity carries no domain, so the call stays `%deferred`, as
/// for [`poisson_process_type`]. Keyword arguments arrive in declared positions.
fn binned_poisson_process_type(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    let open = |t: &Type| {
        matches!(
            t,
            Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_)
        )
    };
    let mut refused = None;
    if let Some((node, bins, _)) = args.first()
        && !open(bins)
        && crate::catalogue::bin_count_shape(bins).is_none()
    {
        inf.diags.push(crate::Diagnostic::error_at(
            *node,
            "`BinnedPoissonProcess`'s `bins` must be bin edges in §07 `bincounts`'s \
             format: a non-empty vector of edges or a record of edge vectors (spec §08)",
        ));
        refused = Some("bins");
    }
    if let Some((node, intensity, _)) = args.get(1)
        && let Some(kind) =
            non_measure_kind(intensity).filter(|_| !matches!(intensity, Type::Kernel { .. }))
    {
        inf.diags.push(crate::Diagnostic::error_at(
            *node,
            format!(
                "`BinnedPoissonProcess`'s `intensity` is {kind}, but spec §08 declares it \
                 a finite-mass measure or kernel over the event space"
            ),
        ));
        refused = Some("intensity");
    }
    // §08 pairs an edge vector with scalar events and a record of edge vectors
    // with record events whose fields hold the bin coordinates.
    if refused.is_none()
        && let (Some((_, bins, _)), Some((node, Type::Measure { domain, .. }, _))) =
            (args.first(), args.get(1))
    {
        let mismatch = match (bins, domain.as_ref()) {
            _ if open(bins) || open(domain) => None,
            (Type::Record(coordinates), Type::Record(fields)) => coordinates
                .iter()
                .find(|(c, _)| fields.iter().all(|(f, _)| f != c))
                .map(|(c, _)| {
                    format!(
                        "has no event field `{}` for the bin coordinate of that name",
                        inf.module.resolve(*c)
                    )
                }),
            (Type::Record(_), _) => Some("has non-record events, but `bins` is a record".into()),
            (_, Type::Scalar(_)) => None,
            _ => Some("has non-scalar events, but `bins` is an edge vector".into()),
        };
        if let Some(why) = mismatch {
            inf.diags.push(crate::Diagnostic::error_at(
                *node,
                format!(
                    "`BinnedPoissonProcess`'s `intensity` {why}: spec §08 bins scalar \
                     events with an edge vector and record events with a record of edge \
                     vectors whose fields match theirs"
                ),
            ));
            refused = Some("intensity");
        }
    }
    if let Some(which) = refused {
        return Type::Failed(
            format!("BinnedPoissonProcess: `{which}` is outside its domain").into(),
        );
    }
    match (args.first(), args.get(1)) {
        (Some((_, bins, _)), Some((_, Type::Measure { .. }, _))) if !open(bins) => {
            match distribution_domain(inf, name, args, named) {
                Some(domain) => Type::Measure {
                    domain: Box::new(domain),
                    mass: Mass::Deferred,
                },
                None => Type::Deferred,
            }
        }
        _ => Type::Deferred,
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
    inf: &Inferencer<'_, '_>,
    node: NodeId,
    output_axes: &[Symbol],
    out: &mut std::collections::HashMap<Symbol, Dim>,
) {
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if inf.is_substituted(node) {
            continue;
        }
        let Node::Call(c) = inf.module.node(node) else {
            continue;
        };
        if let flatppl_core::CallHead::Builtin(op) = c.head {
            let name = inf.module.resolve(op);
            if (name == "get" || name == "get0") && !c.args.is_empty() {
                let arr_ty = inf.node_annotation(c.args[0]).0;
                let flat = flatten_dims(&arr_ty);
                for (k, &idx) in c.args.iter().enumerate().skip(1) {
                    if let Node::Axis(ax) = inf.module.node(idx)
                        && let Some(&d) = flat.get(k - 1)
                    {
                        out.entry(ax.name).or_insert(d);
                    }
                }
                // The first binding fixes each extent. Later body nodes cannot
                // change the output shape once every requested axis has a binding.
                if output_axes.iter().all(|axis| out.contains_key(axis)) {
                    return;
                }
            }
        }
        pending.extend(c.args.iter().rev().copied());
    }
}

/// The phase of `lawof(arg)` — the phase of the **reified law** of `arg`.
///
/// `lawof` absorbs the stochasticity of `draw` ancestors into the law, so the
/// result is deterministic: parametric if the law depends on a free
/// `elementof` leaf, else fixed; never stochastic (spec §04 "Phase of the
/// reified law"). We re-derive the phase over the argument's ancestor subgraph
/// with two overrides vs the normal join: a `draw` contributes the law-phase of
/// its *measure operand* (absorbing the draw), and the recursion bottoms out at
/// `elementof` (parametric) / fixed leaves — mirroring how `functionof`
/// traces to parametric leaves. Ref/`draw` cycles are bounded by `depth`.
fn law_phase(inf: &mut Inferencer<'_, '_>, node: NodeId, depth: u32) -> Phase {
    if depth > 64 {
        return Phase::Parametric; // safe non-stochastic fallback
    }
    match inf.module.node(node).clone() {
        Node::Ref(r) if r.ns == flatppl_core::RefNs::SelfMod => {
            match inf.module.binding_by_name(r.name) {
                Some(b) => {
                    let rhs = inf.module.binding(b).rhs;
                    law_phase(inf, rhs, depth + 1)
                }
                None => Phase::Parametric,
            }
        }
        Node::Call(c) => match c.head {
            flatppl_core::CallHead::Builtin(op) => match inf.module.resolve(op) {
                // Parametric leaf: the law depends on this free input.
                "elementof" => Phase::Parametric,
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
            // parametric (deterministic, may depend on inputs).
            _ => Phase::Parametric,
        },
        // Literals and named constants are fixed; anything else (holes,
        // cross-module refs) is conservatively non-stochastic.
        Node::Lit(_) | Node::Const(_) => Phase::Fixed,
        _ => Phase::Parametric,
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
    Type::Kernel {
        inputs: inputs.into(),
        mass: joint_mass(inf, args, named),
    }
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

/// Direct weighted addition over the ordinary broadcast of weights and inputs.
fn ksuperpose_type(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> RuleStep<Type> {
    if args.len() < 2 {
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            "`ksuperpose` takes the kernel and weights positionally, followed by the kernel inputs",
        ));
        return Ready(Type::Failed("ksuperpose needs a kernel and weights".into()));
    }
    if let Some(failed) = ksuperpose_weights_check(inf, args[1].0, &args[1].1) {
        return Ready(failed);
    }
    if let Some(ty) = refuse_aggregate_operand(
        inf,
        "ksuperpose",
        args[1..]
            .iter()
            .map(|(n, t, _)| (*n, t))
            .chain(named.iter().map(|(_, n, t, _)| (*n, t))),
    ) {
        return Ready(ty);
    }
    let mut shape: Option<Box<[Dim]>> = None;
    for ty in args[1..]
        .iter()
        .map(|(_, t, _)| t)
        .chain(named.iter().map(|(_, _, t, _)| t))
    {
        let axes = match ty {
            Type::Array { shape, .. } => Some(shape.clone()),
            Type::TVector { len, .. } => Some(Box::new([*len]) as Box<[Dim]>),
            Type::Table { nrows, .. } => Some(Box::new([*nrows]) as Box<[Dim]>),
            _ => None,
        };
        if let Some(axes) = axes {
            shape = Some(match shape {
                None => axes,
                Some(previous) => match broadcast_shape_join(&previous, &axes) {
                    Some(joined) => joined,
                    None => {
                        inf.diags.push(crate::Diagnostic::error_at(id,
                            "`ksuperpose` weights and inputs need equal-rank broadcast-compatible collection shapes"));
                        return Ready(Type::Failed(
                            "incompatible ksuperpose broadcast shapes".into(),
                        ));
                    }
                },
            });
        }
    }
    let (weights, weights_ty, weights_phase) = &args[1];
    let weight_shape = match weights_ty {
        Type::Array { shape, .. } => Some(shape.as_ref()),
        Type::TVector { len, .. } => Some(std::slice::from_ref(len)),
        _ => None,
    };
    // Dynamic equal extents need not mean equal sizes at runtime. Prove no
    // expansion only for fixed equal axes, or when weights are the only collection.
    let unexpanded = match (weight_shape, shape.as_deref()) {
        (Some(w), Some(result)) if w == result => {
            w.iter().all(|d| matches!(d, Dim::Static(_)))
                || args[2..]
                    .iter()
                    .map(|(_, t, _)| t)
                    .chain(named.iter().map(|(_, _, t, _)| t))
                    .all(|t| {
                        !matches!(
                            t,
                            Type::Array { .. } | Type::TVector { .. } | Type::Table { .. }
                        )
                    })
        }
        (None, None) => true,
        _ => false,
    };
    fn literal_cells(inf: &Inferencer<'_, '_>, node: NodeId, out: &mut Vec<NodeId>) -> bool {
        let node = resolve_binding_refs(inf, node);
        match inf.module.node(node) {
            Node::Lit(_) => {
                out.push(node);
                true
            }
            Node::Call(call) if matches!(call.head, CallHead::Builtin(op) if inf.module.resolve(op) == "vector") => {
                call.args.iter().all(|&n| literal_cells(inf, n, out))
            }
            Node::Call(call) if matches!(call.head, CallHead::Builtin(op) if matches!(inf.module.resolve(op), "rowstack" | "colstack" | "array")) => {
                call.args
                    .first()
                    .is_some_and(|&n| literal_cells(inf, n, out))
            }
            _ => false,
        }
    }
    let mut literals = Vec::new();
    let literal_unit =
        literal_cells(inf, *weights, &mut literals) && literal_weights_sum_to_one(inf, &literals);
    let mass = ksuperpose_mass(
        ksuperpose_component_mass(inf, args[0].0, &args[0].1),
        weights_ty,
        *weights_phase,
        unexpanded
            && (literal_unit || matches!(inf.lookup_valueset(*weights), ValueSet::StdSimplex(_))),
    );
    let cell = |ty: &Type| match ty {
        Type::Array { elem, .. } | Type::TVector { elem, .. } => *elem.clone(),
        Type::Table { columns, .. } => Type::Record(columns.clone()),
        other => other.clone(),
    };
    let cell_args: Vec<_> = args[2..]
        .iter()
        .map(|(n, t, p)| (*n, cell(t), *p))
        .collect();
    let cell_named: Vec<_> = named
        .iter()
        .map(|(s, n, t, p)| (*s, *n, cell(t), *p))
        .collect();
    ordinary_call_type(inf, args[0].0, &cell_args, &cell_named).map(inf, move |inf, ty| match ty {
        Type::Measure { domain, .. } => Type::Measure { domain, mass },
        Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_) => ty,
        _ => {
            inf.diags.push(crate::Diagnostic::error_at(
                id,
                "`ksuperpose` requires a kernel whose component calls yield measures (spec §06)",
            ));
            Type::Failed("ksuperpose component does not yield a measure".into())
        }
    })
}

/// Broadcast may expand weight axes, but every weight cell is a real scalar.
fn ksuperpose_weights_check(inf: &mut Inferencer<'_, '_>, node: NodeId, ty: &Type) -> Option<Type> {
    fn written_string_cell(inf: &Inferencer<'_, '_>, node: NodeId, depth: usize) -> bool {
        if depth == 0 {
            return false;
        }
        if written_string(inf, node).is_some() {
            return true;
        }
        let node = resolve_binding_refs(inf, node);
        match inf.module.node(node) {
            Node::Call(call) if matches!(call.head, CallHead::Builtin(op) if inf.module.resolve(op) == "vector") => {
                call.args
                    .iter()
                    .any(|&n| written_string_cell(inf, n, depth - 1))
            }
            Node::Call(call) if matches!(call.head, CallHead::Builtin(op) if matches!(inf.module.resolve(op), "rowstack" | "colstack" | "array")) => {
                call.args
                    .first()
                    .is_some_and(|&n| written_string_cell(inf, n, depth - 1))
            }
            _ => false,
        }
    }
    let cell = match ty {
        Type::Array { elem, .. } | Type::TVector { elem, .. } => elem.as_ref(),
        other => other,
    };
    if !written_string_cell(inf, node, 64)
        && matches!(
            cell,
            Type::Scalar(ScalarType::Real | ScalarType::Integer | ScalarType::Boolean)
                | Type::Deferred
                | Type::Any
                | Type::Var(_)
                | Type::Failed(_)
        )
    {
        return None;
    }
    inf.diags.push(crate::Diagnostic::error_at(node,
        "`ksuperpose` weights must have real scalar broadcast cells (spec §06: non-negative weights)"));
    Some(Type::Failed(
        "ksuperpose weight cells are not real scalars".into(),
    ))
}

/// Resolve a bare measure constructor or its alias to the catalogue name.
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

/// Bare constructors carry their declared catalogue mass; reified kernels carry
/// the mass inferred from their body.
fn ksuperpose_component_mass(inf: &Inferencer<'_, '_>, node: NodeId, ty: &Type) -> Mass {
    match component_mass(inf, node, ty) {
        Mass::Unknown => match ksuperpose_constructor_name(inf, node) {
            Some(name) if crate::distribution_param_names(&name).is_some() => {
                crate::builtin_catalogue()
                    .distribution_mass(&name)
                    .unwrap_or(Mass::Unknown)
            }
            Some(name) if name == "Dirac" => Mass::Normalized,
            _ => Mass::Unknown,
        },
        known => known,
    }
}

/// Weighting proper components is generally finite; a unit-sum proof retains normalization.
fn ksuperpose_mass(
    component: Mass,
    weights_ty: &Type,
    weights_phase: Phase,
    simplex: bool,
) -> Mass {
    if component == Mass::Normalized && simplex {
        return Mass::Normalized;
    }
    if component == Mass::Null {
        // Every component is the zero measure, so every weighted term is too.
        return Mass::Null;
    }
    let weights_readable = matches!(
        (weights_ty, weights_phase),
        (
            Type::Array { .. } | Type::TVector { .. } | Type::Scalar(_),
            Phase::Fixed
        )
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

/// `functionof` / `kernelof` (spec §04 reification, §11 reified callables).
/// A `functionof` whose body is a measure *is* a kernel.
///
/// Every admitted reification is fixed: parameters become boundary inputs and
/// draws must be boundary inputs or absorbed into a law.
fn reification_type(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    call: &Call,
    name: &str,
    args: &[ArgInfo],
) -> (Type, Phase) {
    // Boundary entries whose target ref is a `%local` placeholder — that entry
    // IS the placeholder's declaration (spec §04), under either origin tag. The
    // auto-trace itself only ever records `elementof` leaves, but FlatPIR may
    // carry an explicit entry list under `%autoinputs`, and such an entry
    // declares its placeholder exactly as a `%specinputs` entry does.
    let mut declared: Vec<Symbol> = Vec::new();
    // Self-module bindings a boundary entry designates. §04 substitutes each
    // with a fresh `elementof(valueset(a))` input BEFORE the ancestor trace
    // runs, so the referential-transparency walk below cuts there.
    let mut boundary_targets: Vec<Symbol> = Vec::new();
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
                return (
                    Type::Failed("cross-module reification boundary".into()),
                    Phase::Fixed,
                );
            }
            declared.extend(
                entries
                    .iter()
                    .filter(|(_, r)| r.ns == RefNs::Local)
                    .map(|(_, r)| r.name),
            );
            boundary_targets.extend(
                entries
                    .iter()
                    .filter(|(_, r)| r.ns == RefNs::SelfMod)
                    .map(|(_, r)| r.name),
            );
            entries.iter().map(|(n, _)| *n).collect()
        }
        Some(Inputs::Auto) => match inf.auto_inputs_of(id) {
            Some(entries) => {
                declared.extend(
                    entries
                        .iter()
                        .filter(|(_, r)| r.ns == RefNs::Local)
                        .map(|(_, r)| r.name),
                );
                boundary_targets.extend(
                    entries
                        .iter()
                        .filter(|(_, r)| r.ns == RefNs::SelfMod)
                        .map(|(_, r)| r.name),
                );
                entries.iter().map(|(n, _)| *n).collect()
            }
            None => {
                // §04 auto-trace: discover the body's `elementof` parametric
                // leaves (canonical-sorted by name) and fill the side-table, so
                // the reification types as a kernel/function over those inputs.
                let Some((body, _, _)) = args.first() else {
                    return (Type::Deferred, Phase::Fixed);
                };
                let (entries, cross_module) = inf.collect_auto_inputs(*body);
                if cross_module {
                    // Reification is module-local (spec §04): a parametric value
                    // reached through a loaded-module reference cannot become an
                    // input. (A cross-module callable/value may be USED — applied or
                    // referenced — just not taken as a reified input.)
                    inf.diags.push(crate::Diagnostic::error_at(
                        id,
                        "reification is module-local: this depends on a parametric \
                         value from a loaded module, which cannot become an input \
                         (spec §04); use the dependency's callables/values instead, or \
                         reify within the module that defines it",
                    ));
                    return (
                        Type::Failed("cross-module reification".into()),
                        Phase::Fixed,
                    );
                }

                let names: Box<[Symbol]> = entries.iter().map(|(n, _)| *n).collect();
                boundary_targets.extend(
                    entries
                        .iter()
                        .filter(|(_, r)| r.ns == RefNs::SelfMod)
                        .map(|(_, r)| r.name),
                );
                inf.set_auto_inputs(id, entries.into());
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
    let mut seen: HashSet<Symbol> = HashSet::with_capacity(inputs.len());
    let mut repeated_seen: HashSet<Symbol> = HashSet::new();
    let mut repeated: Vec<Symbol> = Vec::new();
    for n in inputs.iter() {
        if !seen.insert(*n) && repeated_seen.insert(*n) {
            repeated.push(*n);
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
        return (
            Type::Failed("repeated boundary input name".into()),
            Phase::Fixed,
        );
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
            return (Type::Failed("undeclared placeholder".into()), Phase::Fixed);
        }
    }
    if matches!(call.inputs, Some(Inputs::Spec(_)))
        && let Some((body, _, _)) = args.first()
        && let Some(parameter) = unbound_parameter(inf, *body, &boundary_targets)
    {
        inf.diags.push(crate::Diagnostic::error_at(
            parameter,
            "an explicit reification boundary must cover every remaining `elementof` \
             dependency; add the parameter or a descendant that cuts its path to the boundary",
        ));
        return (
            Type::Failed("unbound reification parameter".into()),
            Phase::Fixed,
        );
    }
    // Boundaries cut before tracing. kernelof absorbs the remaining draws,
    // while functionof must have no free draw in its reified graph.
    if name == "functionof"
        && let Some((body, _, _)) = args.first()
    {
        let captures = captured_draws(inf, *body, &boundary_targets);
        for (binding, at) in &captures {
            let draw = binding
                .map(|s| format!("draw `{}`", inf.module.resolve(s)))
                .unwrap_or_else(|| "an inline draw".into());
            inf.diags.push(crate::Diagnostic::error_at(
                *at,
                format!(
                    "a reified callable cannot capture {draw}; expose it as a boundary \
                     input, or reify its law with `lawof` or `kernelof`"
                ),
            ));
        }
        if !captures.is_empty() {
            return (Type::Failed("captured draw".into()), Phase::Fixed);
        }
    }
    let body_ty = args.first().map(|(_, t, _)| t);
    let ty = match (name, body_ty) {
        // `kernelof` reifies a value's law after cutting the input boundary.
        // A measure body belongs to functionof; it is not wrapped again.
        (
            "kernelof",
            Some(
                body @ (Type::Measure { .. }
                | Type::Kernel { .. }
                | Type::Function { .. }
                | Type::Likelihood { .. }
                | Type::Tuple(_)
                | Type::Module),
            ),
        ) => {
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
                Type::Kernel { .. } => (
                    "a kernel",
                    "a kernel is already a reified law — pass the value node you meant \
                     to reify, or drop the outer `kernelof`",
                ),
                _ => (
                    "a non-value object",
                    "pass the value node whose law you meant to reify",
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
    };
    (ty, Phase::Fixed)
}

/// Boundary substitution precedes tracing, including for descendant boundaries.
/// Laws absorb draws, but their parameter dependencies still need a boundary.
fn unbound_parameter(
    inf: &Inferencer<'_, '_>,
    body: NodeId,
    boundaries: &[Symbol],
) -> Option<NodeId> {
    let mut pending = vec![body];
    let mut seen = HashSet::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) || inf.lookup_phase(node) == Some(Phase::Fixed) {
            continue;
        }
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                if !boundaries.contains(&r.name)
                    && let Some(binding) = inf.module.binding_by_name(r.name)
                {
                    pending.push(inf.module.binding(binding).rhs);
                }
            }
            Node::Ref(r) if matches!(r.ns, RefNs::Module(_)) => return Some(node),
            Node::Call(c) => {
                if matches!(c.head, CallHead::Builtin(h) if inf.module.resolve(h) == "elementof") {
                    return Some(node);
                }
                pending.extend(inf.module.node(node).children());
            }
            _ => {}
        }
    }
    None
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
    let mut found = Vec::new();
    let mut visited = vec![0u64; module.node_count().div_ceil(64)];
    let mut pending = vec![body];
    while let Some(id) = pending.pop() {
        let word = &mut visited[id.index() / 64];
        let bit = 1u64 << (id.index() % 64);
        if *word & bit != 0 {
            continue;
        }
        *word |= bit;
        match module.node(id) {
            Node::Ref(r) if r.ns == RefNs::Local => {
                if !declared.contains(&r.name) && !found.iter().any(|(p, _)| *p == r.name) {
                    found.push((r.name, id));
                }
                continue;
            }
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                if let Some(b) = module.binding_by_name(r.name) {
                    pending.push(module.binding(b).rhs);
                }
                continue;
            }
            Node::Call(c)
                if c.inputs.is_some()
                    && matches!(c.head, CallHead::Builtin(h)
                        if matches!(module.resolve(h), "functionof" | "kernelof")) =>
            {
                continue;
            }
            _ => {}
        }
        let start = pending.len();
        module.node(id).for_each_child(|child| pending.push(child));
        pending[start..].reverse();
    }
    found
}

/// Every `draw` node the subgraph reified at `body` CAPTURES — reaches without
/// a boundary naming it and without a `lawof` absorbing it — as
/// `(binding name, node)` pairs in walk order. The name is `None` for a
/// `draw(…)` written inline in the reified expression.
///
/// A non-empty result makes the reification invalid. Names identify the free
/// draws that must become boundary inputs or be absorbed by a law.
///
/// `draw` is the only source of a stochastic node (§04 "Phases"; §07 *Random
/// value generation* makes `rand`/`rnginit`/`rngstate` ordinary functions whose
/// "value phases propagate as usual"), so the captured draws are the whole
/// stochastic dependency.
///
/// The walk cuts at four kinds of place, each of which ends the reified
/// subgraph:
///
/// * a boundary target — §04 substitutes it with a fresh
///   `elementof(valueset(a))` input *before* the ancestor trace runs, so
///   `functionof(m, raw_syst = raw_syst)` captures nothing and is the
///   conditional kernel over a declared input. Every reification the walk
///   enters contributes its own boundary names for its own body, so an inner
///   boundary cuts inside the inner body without cutting anywhere else;
/// * a `lawof` call — it "absorbs stochasticity into the reified law rather
///   than propagating it outward" (§04 "Phase of the reified law") — and a
///   `kernelof`, whose body sits under an implicit `lawof` by §04's
///   `functionof(lawof(x), kwargs…)` equivalence;
/// * an `elementof` set argument — §04 traces "back to all leaves of parametric
///   phase — that is, all `elementof` leaves", so the set is outside the
///   subgraph. This mirrors [`Inferencer::collect_auto_inputs`];
/// * any node whose already-inferred phase is **not** stochastic. Phase joins
///   upward, so a `%fixed` or `%parametric` node has no stochastic
///   descendant reaching the output through it. An absent phase does NOT prune
///   — the same "soundly does extra work, never skips" convention
///   `walk_auto` uses.
///
/// Nested reifications are validated at their own construction, including
/// those beneath a law that absorbs the outer graph's stochasticity.
///
/// A cross-module reference is also a cut: a stochastic dependency binding is
/// invisible across the load boundary (`modules.rs` refuses it there), so no
/// draw can reach the subgraph through one.
fn captured_draws(
    inf: &Inferencer<'_, '_>,
    body: NodeId,
    boundary_targets: &[Symbol],
) -> Vec<(Option<Symbol>, NodeId)> {
    let module = &*inf.module;
    let is_draw = |id: NodeId| {
        matches!(module.node(id), Node::Call(c)
            if matches!(c.head, CallHead::Builtin(h) if module.resolve(h) == "draw"))
    };
    // A node whose phase is settled and not stochastic has no stochastic
    // descendant. `None` means the walk has not typed it yet: do not prune.
    let provably_deterministic =
        |id: NodeId| matches!(inf.lookup_phase(id), Some(p) if p != Phase::Stochastic);

    // Boundary scopes, arena-style: entering a reification pushes its own names
    // onto the enclosing scope. `visited` is keyed by (node, scope) because one
    // node can be reached under two scopes and the looser one must still walk.
    let mut scopes: Vec<Vec<Symbol>> = vec![boundary_targets.to_vec()];
    let mut found: Vec<(Option<Symbol>, NodeId)> = Vec::new();
    let mut visited: HashSet<(NodeId, usize)> = HashSet::new();
    let mut pending: Vec<(NodeId, usize)> = vec![(body, 0)];

    while let Some((id, scope)) = pending.pop() {
        if !visited.insert((id, scope)) {
            continue;
        }
        match module.node(id) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                if scopes[scope].contains(&r.name) {
                    continue;
                }
                if let Some(b) = module.binding_by_name(r.name) {
                    let rhs = module.binding(b).rhs;
                    if is_draw(rhs) {
                        // Report the BINDING, not the anonymous `draw` node, so a
                        // caller names the identifier the user wrote.
                        if !found.iter().any(|(n, _)| *n == Some(r.name)) {
                            found.push((Some(r.name), id));
                        }
                    } else if !provably_deterministic(rhs) {
                        pending.push((rhs, scope));
                    }
                }
                continue;
            }
            // A `%local` placeholder is this reification's own input, and a
            // cross-module binding cannot be stochastic (see above).
            Node::Ref(_) => continue,
            Node::Call(c) if c.inputs.is_some() => {
                let head_is =
                    |n: &str| matches!(c.head, CallHead::Builtin(h) if module.resolve(h) == n);
                // `kernelof` absorbs its whole body; and a reification that
                // captures nothing is settled non-stochastic by its own rule.
                if head_is("kernelof") || provably_deterministic(id) {
                    continue;
                }
                // Enter the nested `functionof` under its own boundary names.
                let mut inner = scopes[scope].clone();
                inner.extend(reification_boundary_targets(inf, id));
                inner.sort_unstable();
                inner.dedup();
                let next = match scopes.iter().position(|s| *s == inner) {
                    Some(i) => i,
                    None => {
                        scopes.push(inner);
                        scopes.len() - 1
                    }
                };
                pending.extend(
                    module
                        .node(id)
                        .children()
                        .into_iter()
                        .rev()
                        .map(|c| (c, next)),
                );
                continue;
            }
            Node::Call(c) => {
                if let CallHead::Builtin(h) = c.head {
                    match module.resolve(h) {
                        "lawof" | "elementof" => continue,
                        "draw" => {
                            found.push((None, id));
                            continue;
                        }
                        _ => {}
                    }
                }
                if provably_deterministic(id) {
                    continue;
                }
            }
            _ => continue,
        }
        pending.extend(
            module
                .node(id)
                .children()
                .into_iter()
                .rev()
                .map(|c| (c, scope)),
        );
    }
    found
}

/// The self-module bindings the reification at `id` designates as boundary
/// inputs, under either origin tag. A `%local` placeholder entry names no
/// binding and is skipped: it is the reification's own input, not an ancestor
/// the boundary cuts away.
fn reification_boundary_targets(inf: &Inferencer<'_, '_>, id: NodeId) -> Vec<Symbol> {
    let Node::Call(call) = inf.module.node(id) else {
        return Vec::new();
    };
    let collect = |entries: &[(Symbol, Ref)]| {
        entries
            .iter()
            .filter(|(_, r)| r.ns == RefNs::SelfMod)
            .map(|(_, r)| r.name)
            .collect()
    };
    match call.inputs.as_ref() {
        Some(Inputs::Spec(entries)) => collect(entries),
        Some(Inputs::Auto) => inf.auto_inputs_of(id).map(collect).unwrap_or_default(),
        None => Vec::new(),
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
    /// Record components repeat a field name across their concatenated fields.
    DuplicateField(Symbol),
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
            let mut seen = HashSet::new();
            for t in types {
                let Type::Record(fs) = t else {
                    unreachable!("class 2 is a record")
                };
                for (name, _) in fs {
                    if !seen.insert(*name) {
                        return CatShape::DuplicateField(*name);
                    }
                }
                fields.extend(fs.iter().cloned());
            }
            CatShape::Cat(Type::Record(fields.into()))
        }
    }
}

/// The `cat` type for callers that only want the type (a deferral covers both
/// "mixed" and "unresolved"); callers that diagnose mixing use [`cat_shape`].
pub(crate) fn cat_compose(types: &[Type]) -> Type {
    match cat_shape(types) {
        CatShape::Cat(t) => t,
        CatShape::Mixed | CatShape::DuplicateField(_) | CatShape::Unresolved => Type::Deferred,
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
        CatShape::DuplicateField(field) => {
            let field = inf.module.resolve(field);
            inf.diags.push(crate::Diagnostic::error_at(
                anchor,
                format!(
                    "{what} record components repeat field `{field}`: spec §07 requires distinct field names when concatenating records"
                ),
            ));
            Type::Failed(format!("{what} repeats record field `{field}`").into())
        }
        CatShape::Unresolved => Type::Deferred,
    }
}

/// Check the callable's interface before reading the argument's domain.
fn user_arity_check(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    callee: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    if let Some(sig) = inf
        .module_catalogue_ref(resolve_binding_refs(inf, callee))
        .map(|c| c.sig.clone())
    {
        return module_member_arity_check(inf, id, callee, &sig, args, named);
    }
    let interface = crate::callable::interface(inf, callee)?;
    let reading = arg_reading(inf, args, named, interface.takes_one(), interface.ordered)?;
    let who = match inf.module.node(callee) {
        Node::Ref(r) if r.ns == RefNs::SelfMod => format!("`{}`", inf.module.resolve(r.name)),
        _ => "callable".to_string(),
    };
    if !interface.ordered && !args.is_empty() && !reading.splatting {
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            format!("{who} has keyword-only inputs"),
        ));
        return Some(Type::Failed("keyword-only callable".into()));
    }
    if interface.arity.admits(reading.count) {
        if interface.takes_one()
            && !reading.splatting
            && named.is_empty()
            && let [(arg, ty, _)] = args
        {
            let declared = crate::boundary::declared_input_types(inf, callee);
            if let Some(mismatch) =
                crate::boundary::whole_binding_mismatch(inf, &interface.names, &declared, ty)
            {
                inf.diags.push(crate::Diagnostic::error_at(
                    *arg,
                    format!("{who}: the argument is {mismatch}"),
                ));
                return Some(Type::Failed("argument bound whole".into()));
            }
        }
        return check_double_bound(inf, &interface.names, &who, "§04", args, named)
            .or_else(|| arg_name_check(inf, &interface.names, &who, None, &reading, args, named))
            .or_else(|| {
                missing_required_args(
                    inf,
                    id,
                    &interface.names,
                    interface.arity.min,
                    &reading,
                    args,
                    named,
                )
            });
    }
    let hint = reading.hint();
    let count = interface.arity.min;
    let parameters = if count == 1 {
        "parameter"
    } else {
        "parameters"
    };
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!(
            "{who} declares {count} {parameters}, got {} arguments{hint}",
            reading.count
        ),
    ));
    Some(Type::Failed(
        format!(
            "user call declares {count} {parameters}, got {} arguments",
            reading.count
        )
        .into(),
    ))
}

/// Reject a §09 standard-module member application whose argument count or names
/// contradict the member's catalogue row.
///
/// [`user_arity_check`] used to return `None` here, so no arity or name rule
/// reached a module member at all: `pp.CrystalBall()`, `pp.CrystalBall(1.0)` and
/// `pp.CrystalBall(1.0, 2.0, 3.0, 4.0, 5.0)` all typed as the same normalized
/// `%measure` over `reals` although §09 gives `CrystalBall` four parameters, and
/// `sf.erf(nope = true)` typed the same real scalar as `sf.erf(x)`. §09's
/// Parameters column is the same declaration §08 gives a base distribution, and
/// §04 "Calling conventions" governs the application either way — "All built-in
/// ordinary callables have a defined input order and accept both positional and
/// keyword arguments", and "A call with field or column names that do not match
/// the callable's argument names is a static error" — so a member call is checked
/// exactly as a base call is.
///
/// Deliberately mirrors [`arity_check`]'s sequence rather than sharing its body:
/// that function looks its row up by name in the base catalogue, while a member's
/// row arrives as a `Sig` in the resolution side-table, and the splat guards it
/// runs first are keyed on base-catalogue predicates that no §09 row satisfies —
/// no module row declares an unnamed variadic parameter, and `checked` is a base
/// row. So the two guards are omitted here rather than reimplemented.
///
/// `None` when the row declares no parameter list (`sig_arity` answers `None`), on
/// the same prove-it-is-wrong discipline the base rows use: an undeclared row
/// stays permissive instead of refusing every call to it.
fn module_member_arity_check(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    callee: NodeId,
    sig: &crate::catalogue::Sig,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    let arity = crate::catalogue::sig_arity(sig)?;
    let reading = arg_reading(inf, args, named, arity.admits(1), true)?;
    let got = reading.count;
    let who = format!("`{}`", member_label(inf, callee));
    if arity.admits(got) {
        let names: Vec<String> = crate::catalogue::sig_param_names(sig)?.to_vec();
        if let Some(ty) = check_double_bound(inf, &names, &who, "§09", args, named) {
            return Some(ty);
        }
        return arg_name_check(inf, &names, &who, Some("§09"), &reading, args, named)
            .or_else(|| missing_required_args(inf, id, &names, arity.min, &reading, args, named));
    }
    let declared = arity.describe();
    let hint = reading.hint();
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!("{who} takes {declared} (spec §09), got {got}{hint}"),
    ));
    Some(Type::Failed(
        format!("{who} takes {declared}, got {got}").into(),
    ))
}

/// A §09 member reference as the source spells it: `alias.member`.
fn member_label(inf: &Inferencer<'_, '_>, callee: NodeId) -> String {
    match inf.module.node(callee) {
        Node::Ref(r) => match r.ns {
            RefNs::Module(alias) => format!(
                "{}.{}",
                inf.module.resolve(alias),
                inf.module.resolve(r.name)
            ),
            _ => inf.module.resolve(r.name).to_string(),
        },
        _ => "the module member".to_string(),
    }
}

/// Include record auto-splats and measures nested in argument collections.
fn contains_measure(ty: &Type) -> bool {
    let mut pending = vec![ty];
    while let Some(ty) = pending.pop() {
        match ty {
            Type::Measure { .. } | Type::Kernel { .. } => return true,
            Type::Array { elem, .. } | Type::TVector { elem, .. } => pending.push(elem),
            Type::Record(fields)
            | Type::Table {
                columns: fields, ..
            } => {
                pending.extend(fields.iter().map(|(_, ty)| ty));
            }
            Type::Tuple(parts) => pending.extend(parts.iter()),
            _ => {}
        }
    }
    false
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
) -> RuleStep<Type> {
    let resolved = resolve_binding_refs(inf, callee);
    if let Node::Call(wrapper) = inf.module.node(resolved)
        && let CallHead::Builtin(op) = wrapper.head
    {
        let op = inf.module.resolve(op).to_string();
        let components = wrapper.args.to_vec();
        match op.as_str() {
            "fchain" => return composed_call_type(inf, &components, args, named),
            "broadcasted" if !components.is_empty() => {
                let head = components[0];
                let mut lifted = vec![(
                    head,
                    inf.lookup_type(head).cloned().unwrap_or(Type::Any),
                    Phase::Fixed,
                )];
                lifted.extend_from_slice(args);
                return broadcast_type(inf, resolved, &lifted, named);
            }
            "bijection" if !components.is_empty() => {
                return ordinary_call_type(inf, components[0], args, named);
            }
            "kchain" | "jointchain"
                if components.len() >= 2
                    && matches!(inf.lookup_type(components[0]), Some(Type::Kernel { .. })) =>
            {
                return chain_application_type(inf, &op, &components, args, named);
            }
            _ => {}
        }
    }
    if matches!(inf.module.node(resolved), Node::Ref(r) if matches!(r.ns, RefNs::Module(_)))
        && inf.module_catalogue_ref(resolved).is_none()
        && args
            .iter()
            .map(|(_, ty, _)| ty)
            .chain(named.iter().map(|(_, _, ty, _)| ty))
            .any(contains_measure)
    {
        inf.diags.push(crate::Diagnostic::error_at(
            callee,
            "imported callables with measure-valued arguments are not supported: \
             actual measure substitution across this boundary is not implemented. \
             Apply the helper in its defining module and export the resulting measure, \
             or define the helper in this module",
        ));
        return Ready(Type::Failed("imported measure argument".into()));
    }
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
    if let Some(cref) = inf.module_catalogue_ref(resolved) {
        let note = cref.degraded.clone(); // clone to drop the borrow before &mut inf call
        if let Some(note) = note {
            inf.note_once_str(&note);
        }
        let bound = crate::callable::interface(inf, resolved).and_then(|interface| {
            normalize_arguments(
                inf.module,
                &interface.names,
                args,
                named,
                interface.takes_one(),
                interface.ordered,
            )
        });
        return Ready(catalogue_call_type(
            inf,
            resolved,
            bound.as_deref().unwrap_or(args),
            false,
        ));
    }
    // Prefer the per-call substituted body type (arg types bound to the
    // callable's parameters); fall back to the un-substituted body type for
    // cross-module callables and any case substitution can't bind.
    match callee_ty {
        Type::Function { .. } => {
            substituted_result(inf, callee, args, named, None, true).map(inf, move |inf, sub| {
                sub.map(|(ty, _)| ty)
                    .or_else(|| reified_result_type(inf, callee))
                    .unwrap_or(Type::Deferred)
            })
        }
        // A fan-out kernel (`joint(K1, K2, …)`) is not a reification, so neither
        // substitution nor a reified body reaches its output measure —
        // `kernel_joint_result_type` builds it from the components (spec §06: "At
        // each input point the result is the `joint` of the component output
        // measures").
        Type::Kernel { mass, .. } => {
            let mass = *mass;
            substituted_result(inf, callee, args, named, None, true)
                .map(inf, |_, sub| sub.map(|(ty, _)| ty))
                .map(inf, move |inf, result| {
                    match result
                        .or_else(|| reified_result_type(inf, callee))
                        .or_else(|| kernel_joint_result_type(inf, callee))
                        .or_else(|| tilt::result_type(inf, callee))
                    {
                        Some(Type::Measure { domain, .. }) => Type::Measure { domain, mass },
                        Some(failed @ Type::Failed(_)) => failed,
                        Some(value_ty) => Type::Measure {
                            domain: Box::new(value_ty),
                            mass,
                        },
                        None => Type::Deferred,
                    }
                })
        }
        _ => Ready(Type::Deferred),
    }
}

/// Applying `kchain(K1, K2, ...)` or `jointchain(K1, K2, ...)` with a kernel first
/// (§06 "Dependent composition"): the arguments go to `K1`, and every later kernel
/// gets its fed value whole as its first argument and the same additional arguments
/// (`K2(a, ...)`, `K3([a, b], ...)`). Each step binds them under §04 "Additional
/// arguments". The result is the measure the unapplied chain would be, over its
/// variate.
fn chain_application_type(
    inf: &mut Inferencer<'_, '_>,
    op: &str,
    components: &[NodeId],
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> RuleStep<Type> {
    let Some(fed) = args.first().cloned() else {
        return Ready(Type::Deferred);
    };
    let extras = args[1..].to_vec();
    let components = components.to_vec();
    let (op, named_extras) = (op.to_string(), named.to_vec());
    let first_extras = (extras.clone(), named_extras.clone());
    let (first_args, first_named) =
        match step_arguments(inf, &op, components[0], fed, &first_extras) {
            Ok(bound) => bound,
            Err(refusal) => return Ready(refusal),
        };
    ordinary_call_type(inf, components[0], &first_args, &first_named).map(inf, move |inf, first| {
        if matches!(first, Type::Failed(_)) {
            return first;
        }
        let mut variates = Vec::with_capacity(components.len());
        for (index, &component) in components.iter().enumerate() {
            let ty = inf
                .lookup_type(component)
                .cloned()
                .unwrap_or(Type::Deferred);
            if index > 0
                && (!extras.is_empty() || !named_extras.is_empty())
                && let Some(interface) = crate::callable::interface(inf, component)
                && let Err(refusal) = bind_extras_to_interface(
                    inf,
                    component,
                    &op,
                    component,
                    &interface,
                    1,
                    &extras,
                    &named_extras,
                )
            {
                return refusal;
            }
            match component_variate(inf, component, &ty) {
                Some(variate) => variates.push(variate),
                None => return Type::Deferred,
            }
        }
        let domain = if op == "kchain" {
            variates.last().cloned().unwrap_or(Type::Deferred)
        } else {
            cat_compose(&variates)
        };
        Type::Measure {
            domain: Box::new(domain),
            mass: Mass::Deferred,
        }
    })
}

/// The arguments of one step of a composed callable: the fed value first, then the
/// composed callable's additional arguments bound to the step's remaining inputs
/// (§04 "Additional arguments"). A step with keyword-only inputs takes every
/// argument by name. With no additional arguments the fed value is the only one.
fn step_arguments(
    inf: &mut Inferencer<'_, '_>,
    who: &str,
    step: NodeId,
    fed: ArgInfo,
    extras: &(Vec<ArgInfo>, Vec<NamedInfo>),
) -> Result<(Vec<ArgInfo>, Vec<NamedInfo>), Type> {
    if extras.0.is_empty() && extras.1.is_empty() {
        return Ok((vec![fed], Vec::new()));
    }
    let Some(interface) = crate::callable::interface(inf, step) else {
        return Err(Type::Deferred);
    };
    let mut arguments = vec![fed];
    arguments.extend(bind_extras_to_interface(
        inf, step, who, step, &interface, 1, &extras.0, &extras.1,
    )?);
    if interface.ordered {
        return Ok((arguments, Vec::new()));
    }
    let named = arguments
        .into_iter()
        .zip(&interface.names)
        .map(|((node, ty, phase), input)| (inf.module.intern(input), node, ty, phase))
        .collect();
    Ok((Vec::new(), named))
}

/// Apply composition stages using the ordinary interface before inspecting domains.
///
/// The composed callable passes its additional arguments unchanged to every step
/// (§04 "Function composition": `fchain(f1, f2, f3)(x, ...)` equals
/// `f3(f2(f1(x, ...), ...), ...)`), each later step receiving the previous result
/// whole as its first argument.
fn composed_call_type(
    inf: &mut Inferencer<'_, '_>,
    components: &[NodeId],
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> RuleStep<Type> {
    let Some(fed) = args.first().cloned() else {
        return Ready(Type::Deferred);
    };
    let extras = (args[1..].to_vec(), named.to_vec());
    compose_steps(inf, components, fed, extras)
}

fn compose_steps(
    inf: &mut Inferencer<'_, '_>,
    components: &[NodeId],
    fed: ArgInfo,
    extras: (Vec<ArgInfo>, Vec<NamedInfo>),
) -> RuleStep<Type> {
    let Some((&step, rest)) = components.split_first() else {
        return Ready(Type::Deferred);
    };
    let (args, named) = match step_arguments(inf, "fchain", step, fed, &extras) {
        Ok(bound) => bound,
        Err(refusal) => return Ready(refusal),
    };
    let rest = rest.to_vec();
    ordinary_call_type(inf, step, &args, &named).and_then(inf, move |inf, ty| {
        if rest.is_empty() || matches!(ty, Type::Failed(_)) {
            Ready(ty)
        } else {
            compose_steps(inf, &rest, (step, ty, Phase::Fixed), extras)
        }
    })
}

fn ordinary_call_type(
    inf: &mut Inferencer<'_, '_>,
    callee: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> RuleStep<Type> {
    let normalized = if !args.is_empty() && !named.is_empty() {
        if let Some(interface) = crate::callable::interface(inf, callee) {
            if !interface.ordered {
                inf.diags.push(crate::Diagnostic::error_at(
                    callee,
                    "positional broadcast inputs need an ordered callable interface",
                ));
                return Ready(Type::Failed("unordered positional broadcast inputs".into()));
            }
            if let Some(failed) =
                check_double_bound(inf, &interface.names, "broadcast", "§04", args, named)
            {
                return Ready(failed);
            }
            normalize_arguments(
                inf.module,
                &interface.names,
                args,
                named,
                interface.takes_one(),
                interface.ordered,
            )
        } else {
            None
        }
    } else {
        None
    };
    let (args, named) = match normalized.as_deref() {
        Some(args) => (args, &[][..]),
        None => (args, named),
    };
    let call = Call {
        head: CallHead::User(callee),
        args: args.iter().map(|(node, ..)| *node).collect(),
        named: named
            .iter()
            .map(|(name, node, ..)| flatppl_core::NamedArg {
                kind: flatppl_core::NamedKind::Kwarg,
                name: *name,
                value: *node,
            })
            .collect(),
        inputs: None,
    };
    let ty = inf.lookup_type(callee).cloned().unwrap_or(Type::Any);
    call_rule(
        inf,
        callee,
        &call,
        Some((callee, ty)),
        args,
        named,
        Phase::Fixed,
    )
    .map(inf, |_, (ty, _)| ty)
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
fn catalogue_call_type(
    inf: &mut Inferencer<'_, '_>,
    callee: NodeId,
    args: &[ArgInfo],
    cell: bool,
) -> Type {
    // Clone the sig out to drop the immutable borrow before `lower`'s closures
    // re-borrow `args` inside the `&mut inf` call frame.
    let Some(sig) = inf.module_catalogue_ref(callee).map(|c| c.sig.clone()) else {
        return Type::Deferred;
    };
    // §09 declares a member's parameters as §07 and §08 do, so the same shape rule
    // and the same `%deferred` propagation apply.
    let head = member_label(inf, callee);
    let names = crate::catalogue::sig_param_names(&sig)
        .map(<[String]>::to_vec)
        .unwrap_or_default();
    if let Some(ty) = refuse_shape_outside_sig(
        inf,
        &format!("`{head}`"),
        &head,
        "§09",
        &sig,
        &names,
        args,
        cell,
    ) {
        return ty;
    }
    if matches!(sig, crate::catalogue::Sig::Function { .. })
        && let Some(ty) = unresolved_shaped_argument(&sig, args)
    {
        return ty;
    }
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
    if matches!(sig, crate::catalogue::Sig::Function { .. })
        && unresolved_shaped_argument(&sig, args).is_some()
    {
        return ValueSet::Unknown;
    }
    catalogue_lower(&mut *inf.module, &sig, args).1
}

fn reference_mass(support: &ValueSet) -> Mass {
    match support.is_bounded() {
        Some(true) => Mass::Finite,
        Some(false) => Mass::LocallyFinite,
        None => Mass::Unknown,
    }
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
        written_size: &|i| literal_size(&module.borrow(), args, i),
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
fn pushfwd_codomain(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> RuleStep<Option<Type>> {
    let Some((f_node, _, _)) = args.first() else {
        return Ready(None);
    };
    let mut f_node = *f_node;
    let resolved = resolve_binding_refs(inf, f_node);
    if let Node::Call(call) = inf.module.node(resolved)
        && matches!(call.head, CallHead::Builtin(name) if inf.module.resolve(name) == "bijection")
        && let Some(forward) = call.args.first()
    {
        f_node = *forward;
    }
    let seed = match args.get(1) {
        Some((m_node, Type::Measure { domain, .. }, m_phase))
            if !matches!(domain.as_ref(), Type::Deferred) =>
        {
            let mut inputs = vec![(*m_node, (**domain).clone(), *m_phase)];
            inputs.extend(args.iter().skip(2).cloned());
            Some(inputs)
        }
        _ => None,
    };
    let sub = match seed {
        Some(s) => substituted_result(inf, f_node, &s, &[], None, true),
        None => Ready(None),
    };
    sub.map(inf, move |inf, sub| {
        match sub
            .map(|(ty, _)| ty)
            .or_else(|| reified_result_type(inf, f_node))
        {
            Some(Type::Deferred) | None => None,
            Some(t) => Some(t),
        }
    })
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
pub(crate) fn component_variate(
    inf: &mut Inferencer<'_, '_>,
    node: NodeId,
    ty: &Type,
) -> Option<Type> {
    match ty {
        Type::Measure { domain, .. } => match domain.as_ref() {
            Type::Deferred => None,
            d => Some(d.clone()),
        },
        Type::Kernel { .. } => match reified_result_type(inf, node) {
            Some(Type::Measure { domain, .. }) => match *domain {
                Type::Deferred => None,
                d => Some(d),
            },
            Some(Type::Deferred) => None,
            Some(value_ty) => Some(value_ty),
            None => joint_kernel_variate(inf, node),
        },
        _ => None,
    }
}

/// The output variate of `joint(K1, K2, ...)` or `joint(n1 = K1, ...)` over kernels (spec
/// §06 "Joint measures"): the kernel fans its input out to every component, so its
/// variate is the joint of the components' variates, the `cat` for the positional
/// form and a record naming each one for the keyword form. `None` when `node` is not
/// such a joint or a component's variate is not statically resolvable.
fn joint_kernel_variate(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Option<Type> {
    let node = resolve_binding_refs(inf, node);
    let Node::Call(call) = inf.module.node(node).clone() else {
        return None;
    };
    if !matches!(call.head, CallHead::Builtin(head) if inf.module.resolve(head) == "joint") {
        return None;
    }
    let mut variates = Vec::new();
    for component in call
        .args
        .iter()
        .copied()
        .chain(call.named.iter().map(|n| n.value))
    {
        let ty = inf.lookup_type(component)?.clone();
        variates.push(component_variate(inf, component, &ty)?);
    }
    if call.named.is_empty() {
        Some(cat_compose(&variates))
    } else {
        Some(Type::Record(
            call.named.iter().map(|n| n.name).zip(variates).collect(),
        ))
    }
}

/// `jointchain` output variate (spec §06): the `cat` of every component's
/// variate (positional form, as with `joint`), or a record naming each
/// component's variate (keyword form `jointchain(n1 = …, n2 = …)`). Any
/// component whose variate is not statically resolvable ⇒ `%deferred`.
fn jointchain_domain(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo], named: &[NamedInfo]) -> Type {
    if !named.is_empty() {
        // Keyword form `jointchain(n1 = M, n2 = K, …)` (spec §06): "Each name holds its
        // component's variate as it is, whether scalar, array or record ... A
        // record-valued component becomes a nested record under its name", so the domain is
        // `record{n1: variate(M), n2: variate(K), …}`.
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
/// re-inferring its body in a fresh annotation scope. This is the single-module
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
/// Requests the substituted body's `(type, value-set)` — so a callable whose body
/// tightens its range (`f(x) = sqrt(x)` → `nonnegreals`) carries that set to the
/// call site too. Immediately returns `None` when `callee` is not a local reification, or no
/// parameter could be bound to an argument (caller falls back to the
/// un-substituted body).
fn substituted_result(
    inf: &mut Inferencer<'_, '_>,
    callee: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
    previous: Option<&LocalTargets>,
    retain_targets: bool,
) -> RuleStep<Option<(Type, ValueSet)>> {
    let Some((reif_id, body)) = local_reification(inf, callee) else {
        return Ready(None);
    };
    let Some(inputs) = input_entries(inf, reif_id) else {
        return Ready(None);
    };
    let normalized;
    let args = if let Some(interface) = crate::callable::interface(inf, callee)
        && let Some(bound) = normalize_arguments(
            inf.module,
            &interface.names,
            args,
            named,
            interface.takes_one(),
            interface.ordered,
        ) {
        normalized = bound;
        normalized.as_slice()
    } else {
        args
    };

    // Seed targets: for each parameter bound to a call argument, the body nodes
    // that read it (every matching `%local` placeholder ref, or a self-bound
    // input binding's RHS) annotated with the argument's type/phase/value-set.
    let mut seeds: Vec<(NodeId, crate::modules::Resolved)> = Vec::new();
    let mut local_refs = None;
    let previous = previous.filter(|targets| targets.body == body);
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
        let vset = if inf.lookup_type(arg_node) == Some(&ty) {
            inf.lookup_valueset(arg_node)
        } else {
            ValueSet::natural_of(&ty)
        };
        let res = crate::modules::Resolved {
            ty,
            phase,
            vset,
            result: None,
            set_element: None,
            catalogue: None,
            interface: crate::callable::interface(inf, arg_node),
        };
        match decl.ns {
            RefNs::Local => {
                let targets = previous.unwrap_or_else(|| {
                    local_refs.get_or_insert_with(|| LocalTargets {
                        body,
                        refs: collect_local_refs(inf, body),
                    })
                });
                if let Some(nodes) = targets.refs.get(&decl.name) {
                    seeds.extend(nodes.iter().map(|&node| (node, res.clone())));
                }
            }
            RefNs::SelfMod => {
                if let Some(b) = inf.module.binding_by_name(decl.name) {
                    seeds.push((inf.module.binding(b).rhs, res));
                }
            }
            RefNs::Module(_) => {}
        }
    }
    if seeds.is_empty() {
        return Ready(None);
    }

    // Re-infer ONLY the body with isolated annotations seeded by substitutions.
    // Inferring the body alone (not the whole module via `run`) avoids re-entering
    // the application that triggered this — the seeds cut every parameter, so the
    // body walk never reaches back to the call site.
    RuleStep::InferBody {
        body,
        seeds,
        targets: if retain_targets && inf.level >= Level::Valueset {
            local_refs.map(Box::new)
        } else {
            None
        },
        resume: Box::new(|_, result| Ready(Some(result))),
    }
}

/// Deref a callee expression to its local reification: follow `self` refs to the
/// bound RHS, returning `(reification_node, body_node)` when the RHS is a
/// reification (a call carrying an inputs boundary). `None` otherwise.
pub(crate) fn local_reification(
    inf: &Inferencer<'_, '_>,
    mut node: NodeId,
) -> Option<(NodeId, NodeId)> {
    for _ in 0..inf.module.node_count() {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(c) if c.inputs.is_some() => return Some((node, *c.args.first()?)),
            Node::Call(c) if matches!(c.head, CallHead::User(_)) => {
                node = crate::callable::forwarded_result(inf, node)?;
            }
            _ => return None,
        }
    }
    None
}

/// The ordered input parameters of a reification: `(param-name, declaration-ref)`
/// pairs. A `%specinputs` boundary carries them inline; an `%autoinputs`
/// (keyword-only) boundary reads them from the auto-inputs side-table.
pub(crate) fn input_entries(
    inf: &Inferencer<'_, '_>,
    reif_id: NodeId,
) -> Option<Vec<(Symbol, Ref)>> {
    let Node::Call(call) = inf.module.node(reif_id) else {
        return None;
    };
    match call.inputs.as_ref()? {
        Inputs::Spec(entries) => Some(entries.to_vec()),
        Inputs::Auto => inf.auto_inputs_of(reif_id).map(<[_]>::to_vec),
    }
}

/// Group the body's `%local` placeholder refs by name in traversal order.
/// One call-local walk serves every parameter while seeds retain input order.
///
/// A nested reification that declares the same placeholder rebinds it: §04 binds each
/// `_` to the nearest enclosing `fn`, so `fn(broadcast(fn(_ + 1), _))` has two distinct
/// `_`. The walk skips a name inside the boundary that rebinds it.
fn collect_local_refs(inf: &Inferencer<'_, '_>, body: NodeId) -> HashMap<Symbol, Vec<NodeId>> {
    use std::rc::Rc;
    let mut refs: HashMap<Symbol, Vec<NodeId>> = HashMap::new();
    let mut stack: Vec<(NodeId, Rc<HashSet<Symbol>>)> = vec![(body, Rc::new(HashSet::new()))];
    let mut seen = std::collections::HashSet::new();
    while let Some((id, shadowed)) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match inf.module.node(id) {
            Node::Ref(r) if r.ns == RefNs::Local && !shadowed.contains(&r.name) => {
                refs.entry(r.name).or_default().push(id);
            }
            Node::Call(c) => {
                let inner = match input_entries(inf, id) {
                    Some(entries) if c.inputs.is_some() => {
                        let mut names = (*shadowed).clone();
                        names.extend(
                            entries
                                .iter()
                                .filter(|(_, r)| r.ns == RefNs::Local)
                                .map(|(_, r)| r.name),
                        );
                        Rc::new(names)
                    }
                    _ => Rc::clone(&shadowed),
                };
                if let CallHead::User(callee) = c.head {
                    stack.push((callee, Rc::clone(&shadowed)));
                }
                stack.extend(c.args.iter().map(|&a| (a, Rc::clone(&inner))));
                stack.extend(c.named.iter().map(|n| (n.value, Rc::clone(&inner))));
            }
            _ => {}
        }
    }
    refs
}

/// The first record or tuple among the operands of `broadcast` is a located static error, spec §04
/// "Broadcasting", "Disallowed inputs".
fn refuse_aggregate_operand<'a>(
    inf: &mut Inferencer<'_, '_>,
    who: &str,
    operands: impl Iterator<Item = (NodeId, &'a Type)>,
) -> Option<Type> {
    for (node, ty) in operands {
        let kind = match ty {
            Type::Record(_) => "a record",
            Type::Tuple(_) => "a tuple",
            _ => continue,
        };
        inf.diags.push(crate::Diagnostic::error_at(
            node,
            format!(
                "`{who}` operand is {kind}: records and tuples are not allowed as operands of \
                 `broadcast`, which neither iterates nor holds them constant; a table row \
                 reaching the callable as a record is unaffected (spec §04 \"Broadcasting\", \
                 \"Disallowed inputs\")"
            ),
        ));
        return Some(Type::Failed(format!("{who}: {kind} operand").into()));
    }
    None
}

/// `broadcast(f_or_K, args…)` (spec §04 broadcasting): a deterministic head
/// maps elementwise over same-shape arrays (scalars ride along) into an
/// array; a kernel / distribution-constructor head yields a **measure over
/// the array** of per-cell variates — that is why `draw` of a broadcast
/// distribution produces the observation array.
fn broadcast_type(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> RuleStep<Type> {
    let Some((head_node, head_ty, _)) = args.first() else {
        return Ready(Type::Deferred);
    };
    let (head_node, head_ty) = (*head_node, head_ty.clone());

    // §04 "Broadcasting", "Disallowed inputs": records and tuples are not allowed as
    // operands, neither iterated nor held constant, because broadcasting over them stays
    // open for later. A table row reaching the callable as a record is unaffected: a table
    // operand is a collection of rows, each passed under the callable's own call rule.
    if let Some(ty) = refuse_aggregate_operand(
        inf,
        "broadcast",
        args[1..]
            .iter()
            .map(|(node, ty, _)| (*node, ty))
            .chain(named.iter().map(|(_, node, ty, _)| (*node, ty))),
    ) {
        return Ready(ty);
    }

    if let Some(ty) = pyhf::call_type(inf, id, head_node, &args[1..], named, true) {
        return Ready(ty);
    }

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
                shape = match shape {
                    None => Some(s.clone()),
                    Some(prev) => match broadcast_shape_join(&prev, s) {
                        Some(joined) => Some(joined),
                        None => return Ready(Type::Deferred),
                    },
                };
                elems.push(elem.as_ref().clone());
            }
            Type::Table { columns, nrows } => {
                let rows: Box<[Dim]> = Box::new([*nrows]);
                shape = match shape {
                    None => Some(rows),
                    Some(prev) => match broadcast_shape_join(&prev, &rows) {
                        Some(joined) => Some(joined),
                        None => return Ready(Type::Deferred),
                    },
                };
                elems.push(Type::Record(columns.clone()));
            }
            other => elems.push(other.clone()),
        }
    }
    if let Node::Const(op) = inf.module.node(head_node)
        && matches!(inf.module.resolve(*op), "div" | "mod")
    {
        let name = inf.module.resolve(*op).to_string();
        if let Some(ty) = arity_check(inf, id, &name, &args[1..], named) {
            return Ready(ty);
        }
        let normalized = normalize_keyword_args(inf.module, &name, &args[1..], named);
        let cell = integer_arithmetic_type(inf, &name, normalized.as_deref().unwrap_or(&args[1..]));
        return Ready(match shape {
            Some(shape) if !matches!(cell, Type::Failed(_)) => Type::Array {
                shape,
                elem: Box::new(cell),
            },
            _ => cell,
        });
    }
    let Some(shape) = shape else {
        // No concrete array argument. Broadcasting a DISTRIBUTION is still a
        // measure (an independent product over the eventual broadcast shape), so
        // a reified lambda body `dist.(params)` — whose params are still `%local`
        // placeholders, hence no array yet — must classify as a MEASURE so the
        // reification becomes a kernel rather than a plain function; the
        // call-site substitution then refines the concrete shape. Unknown inputs
        // may still carry collection axes, even when the ordinary scalar call
        // has a known result type.
        let unknown_shape = args[1..]
            .iter()
            .map(|(_, ty, _)| ty)
            .chain(named.iter().map(|(_, _, ty, _)| ty))
            .any(|ty| {
                matches!(
                    ty,
                    Type::Any | Type::Deferred | Type::Var(_) | Type::TVector { .. }
                )
            });
        let cell = |ty: &Type| match ty {
            Type::TVector { elem, .. } => elem.as_ref().clone(),
            ty => ty.clone(),
        };
        let cell_args: Vec<_> = args[1..]
            .iter()
            .map(|(node, ty, phase)| (*node, cell(ty), *phase))
            .collect();
        let cell_named: Vec<_> = named
            .iter()
            .map(|(name, node, ty, phase)| (*name, *node, cell(ty), *phase))
            .collect();
        return ordinary_call_type(inf, head_node, &cell_args, &cell_named).map(
            inf,
            move |_, ty| {
                if !unknown_shape {
                    return ty;
                }
                match ty {
                    Type::Measure { mass, .. } => Type::Measure {
                        domain: Box::new(Type::Deferred),
                        mass,
                    },
                    Type::Failed(_) => ty,
                    _ => Type::Deferred,
                }
            },
        );
    };

    // User-callable head (`broadcast(predict, x = x_data)`): the cell comes from
    // the reified body applied to the PER-CELL argument types — an array input
    // contributes its element type, a scalar rides along — exactly the §09
    // standard-module head treatment below, and the substituted analogue of a
    // direct call.
    let cell_arg = |t: &Type| match t {
        Type::Array { elem, .. } => elem.as_ref().clone(),
        Type::Table { columns, .. } => Type::Record(columns.clone()),
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
            return ordinary_call_type(inf, head_node, &cell_args, &cell_named).map(
                inf,
                move |inf, cell| {
                    if let Type::Record(columns) = cell {
                        if shape.len() != 1 {
                            inf.diags.push(crate::Diagnostic::error_at(
                                id,
                                "record-valued broadcast requires one axis (spec §04)",
                            ));
                            return Type::Failed("multi-axis record broadcast".into());
                        }
                        return Type::Table {
                            columns,
                            nrows: shape[0],
                        };
                    }
                    Type::Array {
                        shape,
                        elem: Box::new(cell),
                    }
                },
            );
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
            return ordinary_call_type(inf, head_node, &cell_args, &cell_named).map(
                inf,
                move |_, cell| {
                    let cell = match cell {
                        Type::Measure { domain, .. } => *domain,
                        value_ty => value_ty,
                    };
                    Type::Measure {
                        domain: Box::new(Type::Array {
                            shape,
                            elem: Box::new(cell),
                        }),
                        mass: broadcast_mass(mass),
                    }
                },
            );
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
    if inf.module_catalogue_ref(head_node).is_some() {
        // A cell is an ordinary call (§04): a table contributes its row record, keyword
        // cells bind by name, and a sole record cell auto-splats.
        let cell_args: Vec<ArgInfo> = args[1..]
            .iter()
            .map(|(n, t, p)| (*n, cell_arg(t), *p))
            .collect();
        let cell_named: Vec<NamedInfo> = named
            .iter()
            .map(|(s, n, t, p)| (*s, *n, cell_arg(t), *p))
            .collect();
        let bound = crate::callable::interface(inf, head_node).and_then(|interface| {
            normalize_arguments(
                inf.module,
                &interface.names,
                &cell_args,
                &cell_named,
                interface.takes_one(),
                interface.ordered,
            )
        });
        let cells = bound.unwrap_or(cell_args);
        return Ready(match catalogue_call_type(inf, head_node, &cells, true) {
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
        });
    }

    // Built-in head: a distribution constructor broadcasts into a measure
    // over the array; a deterministic scalar op maps elementwise.
    let Node::Const(op) = inf.module.node(head_node) else {
        return Ready(Type::Deferred);
    };
    let op_name = inf.module.resolve(*op).to_string();
    if op_name == "record" && args.len() == 1 && !named.is_empty() {
        if shape.len() != 1 {
            inf.diags.push(crate::Diagnostic::error_at(
                id,
                "record-valued broadcast requires one axis (spec §04)",
            ));
            return Ready(Type::Failed("multi-axis record broadcast".into()));
        }
        return Ready(Type::Table {
            columns: named.iter().map(|(n, _, t, _)| (*n, cell_arg(t))).collect(),
            nrows: shape[0],
        });
    }
    if let Some(cell_domain) = distribution_domain(inf, &op_name, &[], &[]) {
        // Each element is an argument to the constructor, so a matrix parameter's
        // element must be a matrix too.
        let cell_args: Vec<ArgInfo> = args[1..]
            .iter()
            .map(|(n, t, p)| (*n, cell_arg(t), *p))
            .collect();
        let cell_named: Vec<NamedInfo> = named
            .iter()
            .map(|(s, n, t, p)| (*s, *n, cell_arg(t), *p))
            .collect();
        let cells = normalize_keyword_args(inf.module, &op_name, &cell_args, &cell_named)
            .unwrap_or(cell_args);
        if let Some(ty) = refuse_argument_outside_declared_shape(inf, &op_name, &cells, true) {
            return Ready(ty);
        }
        return Ready(Type::Measure {
            domain: Box::new(Type::Array {
                shape,
                elem: Box::new(cell_domain),
            }),
            // Independent product of per-cell distributions.
            mass: Mass::Normalized,
        });
    }

    // A §07 collection-domain head (`sum`, `lany`, `l2norm`, `cumsum`, …). §04 applies
    // it to each ELEMENT, so the cell decides: a scalar element misses the head's
    // domain and is refused, an array element is the legal nested case. Placed ahead of
    // the elementwise cell table because both answers are domain rulings, and because
    // the table's `_` arm is what used to swallow this whole family into `%deferred`.
    if let Some((section, domain)) = collection_domain_head(&op_name) {
        let cell = elems.first().cloned().unwrap_or(Type::Deferred);
        if matches!(cell, Type::Scalar(_)) {
            return Ready(refuse_broadcast_collection_cell(
                inf, id, &op_name, section, domain, &cell,
            ));
        }
        if matches!(
            op_name.as_str(),
            "linsolve" | "conv" | "crosscorr" | "cross"
        ) && elems.len() == 2
        {
            let cells: Vec<ArgInfo> = args[1..]
                .iter()
                .map(|(node, ty, phase)| (*node, cell_arg(ty), *phase))
                .collect();
            let cell_named: Vec<NamedInfo> = named
                .iter()
                .map(|(name, node, ty, phase)| (*name, *node, cell_arg(ty), *phase))
                .collect();
            let cells =
                normalize_keyword_args(inf.module, &op_name, &cells, &cell_named).unwrap_or(cells);
            let inner = match op_name.as_str() {
                "linsolve" => linsolve_type(inf, id, &cells),
                "cross" => cross_type(inf, id, &cells),
                _ => convolution_type(inf, id, &op_name, &cells),
            };
            if matches!(inner, Type::Failed(_)) {
                return Ready(inner);
            }
            return Ready(Type::Array {
                shape,
                elem: Box::new(inner),
            });
        }
        return Ready(match broadcast_collection_cell(&op_name, &cell) {
            Some(inner) => Type::Array {
                shape,
                elem: Box::new(inner),
            },
            // An unresolved cell, or a head whose nested answer is deliberately not
            // pinned — never an accusation, since the refusal keys on a cell that IS a
            // scalar rather than on the absence of an answer.
            None => Type::Deferred,
        });
    }

    // The per-cell call of an operator-equivalent head obeys the row's domain: a table's
    // cells are records, and `broadcast(add, t, 1.0)` adds a record to a number.
    // A sole record cell splats onto the row's parameters (`broadcast(sub, table(b = .., a = ..))`).
    let splat_cell = matches!(elems.as_slice(), [Type::Record(fields)]
    if crate::catalogue::builtin().base_param_names(&op_name).is_some_and(|names| {
        fields.iter().all(|(n, _)| names.iter().any(|p| p == inf.module.resolve(*n)))
    }));
    if let Some(domain) = operator_domain(&op_name)
        && !splat_cell
        && let Some(what) = elems
            .iter()
            .find_map(|cell| operand_outside_operator_domain(&op_name, cell))
    {
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            format!(
                "`broadcast` over `{op_name}` has a cell that is {what}, outside the domain \
                 of `{op_name}`: spec §07 \"Operator-equivalent functions\" gives it \
                 \"{domain}\""
            ),
        ));
        return Ready(Type::Failed(
            format!("`{op_name}` broadcast over a cell outside its domain").into(),
        ));
    }

    // Each cell is an ordinary call of the head, so its arguments obey the row's
    // declared shapes: `sqrt.(N)` over a vector of vectors passes `sqrt` a vector.
    let cell_args: Vec<ArgInfo> = args[1..]
        .iter()
        .map(|(n, t, p)| (*n, cell_arg(t), *p))
        .collect();
    let cell_named: Vec<NamedInfo> = named
        .iter()
        .map(|(s, n, t, p)| (*s, *n, cell_arg(t), *p))
        .collect();
    let cells =
        normalize_keyword_args(inf.module, &op_name, &cell_args, &cell_named).unwrap_or(cell_args);
    if let Some(ty) = refuse_argument_outside_declared_shape(inf, &op_name, &cells, true) {
        return Ready(ty);
    }

    let cell = match (op_name.as_str(), elems.as_slice()) {
        ("builtin_logdensityof", [_, _, _]) => Type::Scalar(ScalarType::Real),
        ("add" | "sub" | "mul" | "divide" | "pow" | "min" | "max", [a, b]) => {
            promote2(Some(a), Some(b))
        }
        ("neg", [a]) => a.clone(),
        (
            "exp" | "log" | "sqrt" | "invlogit" | "invprobit" | "logit" | "log1p" | "expm1" | "abs"
            | "sin" | "cos" | "tan" | "atan" | "tanh",
            [a],
        ) => real_or_complex(Some(a)),
        // The comparisons and the logical connectives, whose per-cell result is a
        // boolean whatever the cell kinds were (their catalogue rows all declare
        // `result: Scalar(Boolean)`). This is the ONLY route §07 gives an
        // elementwise comparison — its own Domains column is scalar-only, which
        // `refuse_nonscalar_operand` now enforces — so `v .> 3.0`, `gt.(v, w)` and
        // `broadcast(gt, v, 3.0)` are how a mask is built, and the mask is what
        // §07 "Boolean reductions" hands `lany`/`lall`. Before this the whole
        // dotted family typed `%deferred`, which left that refusal naming a route
        // with no type.
        (
            "equal" | "unequal" | "lt" | "le" | "gt" | "ge" | "land" | "lor" | "lxor" | "lnot"
            | "in" | "isfinite" | "isinf" | "isnan" | "iszero",
            [_] | [_, _],
        ) => Type::Scalar(ScalarType::Boolean),
        // §07 gives `ifelse` a boolean `cond`, so a cell select is the dotted form.
        // The cell is the common type of the branches, as for a direct call.
        ("ifelse", [Type::Scalar(ScalarType::Boolean), a, b]) if a == b => a.clone(),
        ("ifelse", [Type::Scalar(ScalarType::Boolean), a, b]) => promote2(Some(a), Some(b)),
        _ => return Ready(Type::Deferred),
    };
    Ready(Type::Array {
        shape,
        elem: Box::new(cell),
    })
}

/// Join the equal-rank collection shapes accepted by §04 broadcasting.
/// A singleton extent contributes no size constraint, while a dynamic extent
/// keeps the joined extent dynamic.  Two distinct fixed non-singleton extents,
/// or different ranks, have no common broadcast shape.
fn broadcast_shape_join(left: &[Dim], right: &[Dim]) -> Option<Box<[Dim]>> {
    if left.len() != right.len() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(&a, &b)| match (a, b) {
            (x, y) if x == y => Some(x),
            (Dim::Static(1), x) | (x, Dim::Static(1)) => Some(x),
            (Dim::Dynamic, _) | (_, Dim::Dynamic) => Some(Dim::Dynamic),
            (Dim::Static(_), Dim::Static(_)) => None,
        })
        .collect::<Option<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

#[cfg(test)]
mod broadcast_shape_join_tests {
    use super::*;

    /// A zero extent is a valid derived runtime size.  Broadcasting it with a
    /// singleton stays empty; using a numeric maximum here would incorrectly
    /// turn the result into a one-element axis.
    #[test]
    fn zero_with_singleton_stays_zero() {
        assert_eq!(
            broadcast_shape_join(&[Dim::Static(0)], &[Dim::Static(1)]),
            Some(vec![Dim::Static(0)].into_boxed_slice())
        );
    }
}

/// A callable that cannot take one positional argument needs declared names
/// to bind fields from a sole record/table.
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

/// Every argument position holding a size WRITTEN in source: the op, the
/// declared parameter name, its position, and the spec section that floors it
/// positive.
///
/// The written/derived split is the 2026-08-20 owner ruling on zero-size arrays
/// (`flatppl-dev/empty-arrays-ruling.md`, shape C): "an author may not write a
/// degenerate shape, data may turn out empty". So this table is exactly the
/// sizes an author writes; §06's `iid` entry states the other half in the spec
/// itself — "a `size` derived from data rather than written in source may
/// resolve to 0, giving the empty product measure".
///
/// NOT here: `addaxes`'s two counts, which §07 declares non-negative (0 axes
/// inserted is a legal no-op), and index positions such as `onehot`'s `i` or
/// `get`'s selector, which are a separate 1-based-indexing rule.
const WRITTEN_SIZE_ARGS: &[(&str, &str, usize, &str)] = &[
    ("array", "size", 1, "§07 \"Array and table generation\""),
    ("fill", "size", 1, "§07 \"Array and table generation\""),
    ("zeros", "size", 0, "§07 \"Array and table generation\""),
    ("ones", "size", 0, "§07 \"Array and table generation\""),
    ("eye", "n", 0, "§07 \"Array and table generation\""),
    ("onehot", "n", 1, "§07 \"Array and table generation\""),
    ("linspace", "n", 2, "§07 \"Array and table generation\""),
    ("extlinspace", "n", 2, "§07 \"Array and table generation\""),
    ("tile", "size", 1, "§07 \"Array and table operations\""),
    (
        "splitblocks",
        "blocksize",
        1,
        "§07 \"Array and table operations\"",
    ),
    ("partition", "spec", 1, "§07 \"Array and table operations\""),
    ("bandedmat", "rows", 1, "§07 \"Array and table operations\""),
    ("cartpow", "size", 1, "§03 \"Cartesian power\""),
    ("stdsimplex", "n", 0, "§03 \"Standard simplex\""),
    ("ordered", "n", 0, "§03 \"Ordered vectors\""),
    ("zerosum", "n", 0, "§03 \"Zero-sum vectors\""),
    ("iid", "size", 1, "§06 `iid`"),
    ("markovchain", "n", 2, "§06 `markovchain`"),
    ("LKJ", "n", 0, "§08 `LKJ`"),
    ("LKJCholesky", "n", 0, "§08 `LKJCholesky`"),
];

/// Refuse a non-positive size written in source (see [`WRITTEN_SIZE_ARGS`]).
///
/// Only the WRITTEN spelling is read — an integer literal, a negated one, or a
/// literal inside a written size vector. A size computed from data is derived
/// and stays legal: it may resolve to 0, and the ruling's sub-ruling 3 keeps
/// that so whether or not the const-eval table can fold it, "legality must not
/// depend on optimizer strength".
fn written_size_check(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    let (_, param, pos, section) = WRITTEN_SIZE_ARGS.iter().find(|(op, ..)| *op == name)?;
    let (size_node, size_ty) = named
        .iter()
        .find(|(n, ..)| inf.module.resolve(*n) == *param)
        .map(|(_, n, t, _)| (*n, t.clone()))
        .or_else(|| args.get(*pos).map(|(n, t, _)| (*n, t.clone())))?;
    // A size is an integer, or a vector of them, by TYPE: `fill(0.0, 2.0)` writes
    // a real, whatever number it holds. A size the catalogue types as a scalar
    // integer (`eye`'s `n`) is left to the catalogue's own type and shape checks.
    let typed_by_catalogue = crate::catalogue::builtin()
        .base(name)
        .is_some_and(|sig| crate::catalogue::sig_param_integral(sig, pos + 1)[*pos].is_some());
    let elem = match &size_ty {
        _ if typed_by_catalogue => &Type::Any,
        Type::Array { shape, elem } if shape.len() == 1 => elem.as_ref(),
        other => other,
    };
    if let Type::Scalar(got @ (ScalarType::Real | ScalarType::Complex)) = elem {
        let subject = format!(
            "`{name}`'s `{param}`{} is",
            if elem == &size_ty { "" } else { " vector" }
        );
        let message = non_integral_message(
            &subject,
            *got,
            section,
            crate::catalogue::ScalarTag::Integer,
        );
        inf.diags
            .push(crate::Diagnostic::error_at(size_node, message));
        return Some(Type::Failed(format!("{name}: real-typed size").into()));
    }
    // A written size vector contributes one axis per element; a scalar size is
    // the single axis. Either way the elements are read as written.
    let axes: Vec<NodeId> = match inf.module.node(size_node).clone() {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(op) if inf.module.resolve(op) == "vector") => {
            c.args.to_vec()
        }
        _ => vec![size_node],
    };
    let multi = axes.len() > 1;
    let mut refused = false;
    for (i, axis) in axes.iter().enumerate() {
        let Some(n) = written_int(inf, *axis) else {
            continue;
        };
        if n > 0 {
            continue;
        }
        let which = if multi {
            format!("axis {} of `{name}`'s `{param}`", i + 1)
        } else {
            format!("`{name}`'s `{param}`")
        };
        inf.diags.push(crate::Diagnostic::error_at(
            *axis,
            format!(
                "{which} is written as `{n}`, but spec {section} requires a \
                 positive size. A size computed from data may resolve to 0 — \
                 giving an empty array, or the empty product measure — but a \
                 size written in source may not."
            ),
        ));
        refused = true;
    }
    refused.then(|| Type::Failed(format!("{name}: non-positive written size").into()))
}

/// §07 `linspace`: "When `n = 1`, `from` and `to` must be equal." `extlinspace` is
/// `linspace` with two overflow edges, so the same rule holds. Only written values
/// are compared, as for [`written_size_check`].
fn single_point_linspace_check(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    if !matches!(name, "linspace" | "extlinspace") {
        return None;
    }
    let node = |key: &str, pos: usize| named_or_positional_node(inf.module, named, args, key, pos);
    let (from, to, n) = (node("from", 0)?, node("to", 1)?, node("n", 2)?);
    if written_int(inf, n) != Some(1) {
        return None;
    }
    let (from, to) = (written_scalar(inf, from)?, written_scalar(inf, to)?);
    if from == to {
        return None;
    }
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!(
            "`{name}` with `n = 1` has the single point `from`, so `from` and `to` must be \
             equal (spec §07 `linspace`); they are written as `{from}` and `{to}`"
        ),
    ));
    Some(Type::Failed(
        format!("{name}: one point between unequal ends").into(),
    ))
}

/// The value of an integer size WRITTEN at this node: a literal, or `neg` over
/// one (how the lexer spells `-3`). Anything else is not a written size.
fn written_int(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<i64> {
    match inf.module.node(node) {
        Node::Lit(Scalar::Int(n)) => Some(*n),
        Node::Call(c) => {
            let CallHead::Builtin(op) = c.head else {
                return None;
            };
            if inf.module.resolve(op) != "neg" {
                return None;
            }
            let inner = *c.args.first()?;
            match inf.module.node(inner) {
                Node::Lit(Scalar::Int(n)) => n.checked_neg(),
                _ => None,
            }
        }
        _ => None,
    }
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
    let reading = arg_reading(inf, args, named, cat.base_takes_aggregate_whole(name), true)?;
    let got = reading.count;
    // A splat onto a row whose variadic inputs are UNNAMED can never bind, whatever the
    // count, so this precedes the arity comparison: with a fitting column count the arity
    // would pass and the nameless row would then ACCEPT, and with a non-fitting one the
    // arity message would describe a count problem instead of the real one.
    if reading.splatting && cat.base_has_unnamed_variadic(name) {
        return Some(refuse_splat_onto_unnamed_variadic(inf, id, name, cat, args));
    }
    if arity.admits(got) {
        // Positional calls cannot introduce unknown names or double bindings.
        if named.is_empty() && !reading.splatting {
            return None;
        }
        // The count is right; the names still have to be the declared ones. `?`
        // here means "this row declares none — accept the call", not a failure to
        // propagate; see `base_param_names` for which rows are nameless and why.
        let names = cat.base_param_names(name)?.to_vec();
        let section = cat.base_param_section(name);
        let who = format!("`{name}`");
        if let Some(ty) = check_double_bound(inf, &names, &who, section, args, named) {
            return Some(ty);
        }
        return arg_name_check(inf, &names, &who, Some(section), &reading, args, named)
            .or_else(|| missing_required_args(inf, id, &names, arity.min, &reading, args, named));
    }
    // Same section mapping as the name check below, so a row documented outside §07 —
    // `bijection`, `logdensityof` — cites §06 in BOTH its diagnostics rather than only one.
    let section = cat.base_param_section(name);
    let declared = arity.describe();
    // `got` is the SPLAT count on a splatting call, so the author sees a number
    // larger than the arguments they wrote — say where it came from.
    let hint = reading.hint();
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!("`{name}` takes {declared} (spec {section}), got {got}{hint}"),
    ));
    Some(Type::Failed(
        format!("{name} takes {declared}, got {got}").into(),
    ))
}

/// Reject an aggregate constructor whose §03 well-formedness the per-op rules do
/// not check: a duplicate field or column name, a positional-and-keyword mix on a
/// set or measure product, or an array literal that mixes a string element with a
/// value element.
///
/// Each of the three lets an ill-formed model type cleanly and reach the
/// determiniser: `record(a = 1, a = 2)` produced a two-field record both of whose
/// fields are named `a`, `cartprod(reals, a = integers)` dropped its positional
/// component from the value set, and `[1, "x"]` typed an integer array of integers.
fn constructor_check(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    match name {
        // §03 "Cartesian product" defines the two spellings SEPARATELY and gives
        // each its own member type: `cartprod(S1, S2, ...)` yields "a set of
        // arrays, not a set of tuples", while "The keyword form `cartprod(a = S1,
        // b = S2, ...)` produces a set of records". No sentence assigns a meaning
        // to a call that writes both, so the mixed form has no member type to
        // give. `joint` already refused it (`refuse_mixed_joint_spelling`); this
        // extends the same refusal to the set product §03 says `joint` mirrors,
        // and to `jointchain`, which §04 lists alongside them.
        //
        // Unrefused, `set_call_valueset_at` took the named branch whenever ANY
        // named argument was present and silently dropped every positional one:
        // `cartprod(reals, a = integers)` had value set `record(a = integers)` and
        // `elementof` of it yielded only the record field.
        "cartprod" | "jointchain" if !args.is_empty() && !named.is_empty() => {
            Some(refuse_mixed_product_spelling(inf, id, name))
        }
        "record" => {
            if let Some((field, node, _, _)) = named
                .iter()
                .find(|(_, _, ty, _)| matches!(ty, Type::Module))
            {
                inf.diags.push(crate::Diagnostic::error_at(
                    *node,
                    format!(
                        "record field `{}` is a module, but modules are namespaces rather than values and cannot be stored in a record (spec §11 Type categories)",
                        inf.module.resolve(*field)
                    ),
                ));
                Some(Type::Failed("module-valued record field".into()))
            } else {
                refuse_duplicate_field_names(inf, name, named)
            }
        }
        "table" | "joint" | "jointchain" | "cartprod" => {
            refuse_duplicate_field_names(inf, name, named)
        }
        "vector" => refuse_mixed_string_array(inf, args),
        "cat" => refuse_cat_string_vectors(inf, id, args),
        _ => None,
    }
}

/// Reject concatenation between a written string-selector vector and a value
/// vector. Their element `Type`s alone cannot distinguish strings from `%any`.
fn refuse_cat_string_vectors(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    args: &[ArgInfo],
) -> Option<Type> {
    let kinds: Vec<(NodeId, bool)> = args
        .iter()
        .filter_map(|(node, _, _)| written_vector_is_string(inf, *node).map(|k| (*node, k)))
        .collect();
    if !kinds.iter().any(|(_, strings)| *strings) || !kinds.iter().any(|(_, strings)| !*strings) {
        return None;
    }
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        "`cat` cannot concatenate a string selector list with a value vector: spec §07 admits vectors with one element domain, and strings are not scalar values"
            .to_string(),
    ));
    Some(Type::Failed(
        "cat mixes a string selector list and a value vector".into(),
    ))
}

/// Classify a written vector as all strings or all known values.
fn written_vector_is_string(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<bool> {
    let mut node = node;
    for _ in 0..64 {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(call) if matches!(call.head, CallHead::Builtin(head) if inf.module.resolve(head) == "vector") =>
            {
                let mut kind = None;
                for &child in call.args.iter() {
                    let child_kind = if written_string(inf, child).is_some() {
                        true
                    } else if matches!(
                        inf.lookup_type(child),
                        Some(Type::Scalar(_) | Type::Array { .. } | Type::TVector { .. })
                    ) {
                        false
                    } else {
                        return None;
                    };
                    if kind.is_some_and(|prior| prior != child_kind) {
                        return None;
                    }
                    kind = Some(child_kind);
                }
                return kind;
            }
            _ => return None,
        }
    }
    None
}

/// Reject a set or measure product written with both positional and keyword
/// components. Mirrors [`refuse_mixed_joint_spelling`], which owns `joint`.
fn refuse_mixed_product_spelling(inf: &mut Inferencer<'_, '_>, id: NodeId, name: &str) -> Type {
    let (kind, produces) = if name == "cartprod" {
        (
            "set",
            "a set of arrays (positional) or a set of records (keyword)",
        )
    } else {
        (
            "measure",
            "an array-variate measure (positional) or a record-variate one (keyword)",
        )
    };
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!(
            "`{name}` mixes positional and keyword components: spec §03 \"Cartesian \
             product\" defines the two spellings separately, and each produces a \
             different {kind} — {produces}. A mixed call has neither. Name every \
             component, or name none"
        ),
    ));
    Type::Failed(format!("{name} mixes positional and keyword components").into())
}

/// Reject a constructor that declares the same field, column or component name
/// twice.
///
/// §03 "Records" makes the name the only access path — "Fields are accessed by
/// name, not by position — `get(r, i)` is not supported to avoid ambiguity with
/// row indexing on tables" — so a second field of the same name is unreachable,
/// and which of the two `r.a` denotes is undefined. §03 "Tables" gives columns the
/// same access path ("Column access by field name: `t.colname`"), and a duplicate
/// there is unreachable for the same reason. A keyword `joint` or `cartprod`
/// carries the names into a record-variate measure or a record set, where they hit
/// exactly the same wall.
///
/// Unrefused, all four kept the duplicate: `record(a = 1, a = 2)` typed
/// `(%record (a (%scalar integer)) (a (%scalar integer)))`, and the joint's domain
/// carried two `a` fields into the determiniser's density record.
///
/// Anchored at the SECOND occurrence, which is the one the author can delete.
fn refuse_duplicate_field_names(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    named: &[NamedInfo],
) -> Option<Type> {
    let mut seen: HashSet<Symbol> = HashSet::with_capacity(named.len());
    let mut first_dup = None;
    for (sym, node, _, _) in named {
        if !seen.insert(*sym) {
            let field = inf.module.resolve(*sym).to_string();
            inf.diags.push(crate::Diagnostic::error_at(
                *node,
                format!(
                    "`{name}` declares `{field}` twice: spec §03 \"Records\" accesses \
                     fields by name, not by position, so a repeated name is \
                     unreachable and `.{field}` has no defined component"
                ),
            ));
            first_dup = first_dup.or(Some(field));
            continue;
        }
    }
    first_dup.map(|f| Type::Failed(format!("{name} declares `{f}` twice").into()))
}

/// Reject an array literal that mixes a string element with a value element.
///
/// §03 "Arrays" admits two kinds of element: "collections of scalar values (real,
/// integer, boolean and complex values) or arrays". A string is neither, and §03
/// "Scalar types" does not list one — its only role is as a field-name selector,
/// which the spec always writes as a HOMOGENEOUS list (§06
/// `disintegrate(["obs"], …)`, §04 `relabel(x, ["x", "y", "z"])`). So a
/// homogeneous string array is left alone, and a mixed one is refused: it is
/// neither an array of values nor a selector list.
///
/// The check reads the argument NODES, not their types, because a string carries
/// no `Type` — [`literal_type`] answers `%any` for one. That is what made the
/// defect invisible: the element join absorbed the `%any` into its numeric
/// neighbour, so `[1, "x"]`, `["x", 1]` and even `[true, "x"]` all typed as
/// `(%array 1 (2) (%scalar integer))` with value set `(cartpow integers 2)` — an
/// integer element type for two elements, one of which is not a number.
fn refuse_mixed_string_array(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Option<Type> {
    let strings: Vec<bool> = args
        .iter()
        .map(|(node, _, _)| written_string(inf, *node).is_some())
        .collect();
    if !strings.iter().any(|s| *s) || strings.iter().all(|s| *s) {
        return None;
    }
    for ((node, _, _), is_string) in args.iter().zip(&strings) {
        if !is_string {
            continue;
        }
        inf.diags.push(crate::Diagnostic::error_at(
            *node,
            "an array element is a string, but a sibling element is a value: spec §03 \
             \"Arrays\" admits \"scalar values (real, integer, boolean and complex \
             values) or arrays\" as elements, and a string is neither. A string array \
             is a field-name selector list, which holds only names"
                .to_string(),
        ));
    }
    Some(Type::Failed("array mixes strings and values".into()))
}

/// How §04 "Calling conventions" declares a special operation's inputs. The
/// catalogue carries no row for these heads — `ops.rs` types them structurally —
/// so their arity lives here, read off §04's enumerated list rather than derived.
///
/// §04's framing supplies the axis: "Special operations have zero to three
/// *distinguished inputs*: unnamed, ordered inputs of fixed arity. They may have
/// additional variadic named or unnamed inputs."
///
/// **A distinguished input is POSITIONAL ONLY, and §04 now says so outright:**
///
/// > A distinguished input has no name and so cannot be passed by keyword. The
/// > measure combinators likewise take their inputs positionally: a keyword
/// > spelling such as `normalize(M = mu)` is a static error. Where this
/// > specification refers to a distinguished input by a name, as in
/// > `aggregate(f_reduction, output_axes, expr)`, the name identifies the input in
/// > prose only. A call binds the input by position, never by keyword argument.
///
/// The last two sentences arrived with flatppl-design PR #109 (`c44993f`), which
/// settled the question this code was written against. Before it, the rule had to
/// be READ off §04's own `normalize(M)`: that entry names its input in exactly the
/// same signature-plus-bullets shape and §04 declared its keyword spelling an
/// error, so the shape could not be what confers keyword binding. The spec now
/// states the conclusion directly.
///
/// **Adjudicated 2026-09-03** —
/// `flatppl-dev/adjudication-keyword-distinguished-inputs.md`, user-ratified, and
/// merged as PR #109. An
/// earlier revision of this table checked counts only and left both spellings as
/// it found them, because the crate carried two readings: `ksuperpose_type`
/// refused the keyword form citing §04 while `aggregate` accepted it. That split
/// was option C of the adjudication — "keyword allowed only where a later section
/// names the input" — and it was REJECTED: it is two rules for one concept, and
/// "names the input" is not a test a user can apply, since `ksuperpose(kernel,
/// weights)` has names but no bullets and `normalize(M)` has a name and is
/// forbidden by example. Option B, keyword everywhere, was rejected too: it needs
/// a normative name for inputs that have none, and it collides with the
/// user-chosen keyword namespace at `functionof`, `kernelof`, `broadcast` and
/// `load_module`.
///
/// `load_data` is deliberately absent from this table and unaffected: the
/// adjudication finds it mis-classified — it has no template, binding scope or
/// variadic part — and reclassified it as an ordinary builtin in its own spec PR,
/// flatppl/flatppl-design#110, which deleted its §04 bullet. Its arity comes from
/// the catalogue row instead; `special_arity::load_data_is_an_ordinary_builtin_not_a_special_operation`
/// in `tests/special_arity.rs` pins that.
enum SpecialArity {
    /// Exactly `n` distinguished inputs, positional only. `standard_module` is
    /// no longer a separate variant: §04 "Standard modules" says of it outright
    /// that it "only accepts positional arguments, not keyword arguments", which
    /// is now what EVERY distinguished input does, so the carve-out folded in.
    Distinguished(usize),
    /// `n` distinguished POSITIONAL inputs, then zero or more additional
    /// arguments, positional or keyword, that bind the callback's remaining
    /// inputs (§04 "Additional arguments").
    DistinguishedPlusTail(usize),
    /// `n` distinguished inputs plus optional variadic NAMED inputs
    /// (`load_module`'s substitutions).
    DistinguishedPlusNamed(usize),
    /// `n` distinguished POSITIONAL inputs, then variadic inputs that may be
    /// named or unnamed — `broadcast`'s shape. §04: "`broadcast`: One
    /// distinguished input for the function to be broadcast, plus named or
    /// unnamed inputs that match the inputs of that function."
    ///
    /// Distinct from [`SpecialArity::VariadicEither`], which has no distinguished
    /// input and so admits an all-keyword call: `joint(a = M1, b = M2)` is a
    /// spec-sanctioned spelling, while `broadcast(f = Poisson, rate = v)` is not.
    /// That call typed `%deferred` at exit 0 before this variant existed —
    /// `broadcast_type` reads the function from position 0, found nothing there,
    /// and deferred, where the positional spelling typed a normalized `%measure`.
    DistinguishedPlusEither(usize),
    /// Variadic unnamed inputs, at least `n`.
    VariadicPositional(usize),
    /// Variadic inputs that may be unnamed OR named, at least `n` in total.
    VariadicEither(usize),
    /// Variadic NAMED inputs, at least `n` — and, for `record`/`table`, the one
    /// positional argument §03 "Tables" gives their record–table conversions
    /// ("Tables can also be constructed from records of equal-length vectors via
    /// `table(r)` and converted to such records via `record(t)`").
    VariadicNamedOrOneAggregate(usize),
}

/// The §04 input declaration of special operation `name`, with the bullet that
/// states it, or `None` for every head §04's list omits.
///
/// Every entry below is one line of §04 "Calling conventions". The heads §04 lists
/// that are ALSO base catalogue rows — `vector`, `cat`, `get`, `get0`,
/// `load_data`, `checked` — are deliberately absent: [`arity_check`] answers them
/// from the row and returns before this table is consulted, and duplicating the
/// count here would give one rule two sources. `functionof` and `kernelof` are
/// absent for a different reason: a reification is typed on the
/// `call.inputs.is_some()` path, which returns above this check. `ksuperpose` is
/// absent for a third: [`ksuperpose_type`] already implements the same §04 rule
/// with a message naming its two inputs, so a second copy here would report one
/// defect twice in different words.
fn special_arity(name: &str) -> Option<(SpecialArity, &'static str)> {
    use SpecialArity::*;
    let entry = match name {
        "elementof" | "external" | "draw" => (
            Distinguished(1),
            "\"`elementof`, `external`, `draw`: One distinguished input\"",
        ),
        "lawof" | "fixed" => (
            Distinguished(1),
            "\"`lawof`, `fixed`: One distinguished input\"",
        ),
        "broadcasted" => (
            Distinguished(1),
            "\"`broadcasted`: One distinguished input\"",
        ),
        "standard_module" => (
            Distinguished(2),
            "\"`standard_module`: Two distinguished inputs\", and §04 \"Standard \
             modules\": \"`standard_module` only accepts positional arguments, not \
             keyword arguments\"",
        ),
        "tilt" => (
            Distinguished(2),
            "§06 `tilt(M, B)` takes two positional arguments",
        ),
        "aggregate" | "metricsum" => (
            Distinguished(3),
            "\"`aggregate`, `metricsum`: Three distinguished inputs\"",
        ),
        "markovchain" => (
            DistinguishedPlusTail(3),
            "\"`markovchain`: Three distinguished inputs, plus variadic inputs\"",
        ),
        "scan" | "kscan" => (
            DistinguishedPlusEither(2),
            "\"`scan`, `kscan`: Two distinguished inputs, plus one or more variadic sequence inputs\"",
        ),
        "weighted" | "logweighted" | "pushfwd" | "reduce" | "filter" => (
            DistinguishedPlusTail(2),
            "two distinguished inputs and additional arguments for the callback (§04 \"Additional arguments\")",
        ),
        "load_module" => (
            DistinguishedPlusNamed(1),
            "\"`load_module`: One distinguished input plus optional variadic named \
             inputs with no significant order\"",
        ),
        "fchain" | "kchain" => (
            VariadicPositional(1),
            "\"`cat`, `fchain`, `kchain`: Variadic unnamed inputs with significant order\"",
        ),
        "superpose" => (
            VariadicPositional(1),
            "\"`superpose`: Variadic unnamed inputs with no significant order\"",
        ),
        "tuple" => (
            VariadicPositional(2),
            "\"`tuple`: Unnamed variadic inputs with significant order (minimum two)\"",
        ),
        "cartprod" | "joint" | "jointchain" => (
            VariadicEither(1),
            "\"`cartprod`, `joint`, `jointchain`: Variadic unnamed or named inputs \
             with significant order\"",
        ),
        "broadcast" => (
            DistinguishedPlusEither(1),
            "\"`broadcast`: One distinguished input for the function to be broadcast, \
             plus named or unnamed inputs that match the inputs of that function\"",
        ),
        "record" | "table" => (
            VariadicNamedOrOneAggregate(1),
            "\"`record` and `table`: Named variadic inputs with significant order\"",
        ),
        _ => return None,
    };
    Some(entry)
}

/// Heads whose leading inputs cannot be passed by keyword although they carry a
/// catalogue row, as `(distinguished positional inputs, keywords allowed after
/// them, citation)`. §04 "Calling conventions" names the measure combinators
/// (`normalize(M = mu)` is its example); §06 lists them. `restrict` keeps its
/// keyword-splat form after the measure (§06 "Measure restriction"). `cat` and
/// `get` are listed in §04 with unnamed variadic inputs.
fn positional_only_inputs(name: &str) -> Option<(usize, bool, &'static str)> {
    const MEASURE_COMBINATORS: &str = "\"The measure combinators likewise take their non-variadic \
         inputs positionally: a keyword spelling such as `normalize(M = mu)` is a static error\"";
    let entry = match name {
        "normalize" | "totalmass" => (1, false, MEASURE_COMBINATORS),
        "iid" | "truncate" | "bayesupdate" | "likelihoodof" | "disintegrate" => {
            (2, false, MEASURE_COMBINATORS)
        }
        "locscale" | "bijection" => (3, false, MEASURE_COMBINATORS),
        "restrict" => (1, true, MEASURE_COMBINATORS),
        "cat" => (
            0,
            false,
            "\"`cat`, `fchain`, `kchain`: Variadic unnamed inputs\"",
        ),
        "get" => (
            1,
            false,
            "\"`get`: One distinguished input plus variadic unnamed input\"",
        ),
        _ => return None,
    };
    Some(entry)
}

/// Reject a special-operation application whose input count or spelling
/// contradicts §04's declaration for it, and any nullary builtin call.
///
/// Unenforced, `aggregate()`, `metricsum()`, `load_module()`, `standard_module()`,
/// `record()`, `table()` and `broadcast()` all typed at exit 0 — `record()` as an
/// empty record and the two loaders as `%module` — although §04 states both the
/// flat rule ("Nullary calls (`f()`) are not allowed.") and, for the special
/// operations specifically, "The total number of inputs is never zero". Over-supply
/// passed the same way: `standard_module("particle-physics", "0.1", "ignored")`
/// typed `%module` with the third argument discarded.
///
/// The nullary refusal is the fallback and applies to every builtin head that
/// reaches here, not only the ones [`special_arity`] tabulates: a head with a
/// catalogue row has already been answered by [`arity_check`], so anything still
/// arriving with no arguments at all is a head whose declared count is unknown —
/// and §04's rule needs no count to apply.
///
/// **`joint()` and its three siblings are included, with no exemption** (owner
/// ruling). An earlier revision exempted `joint`, `jointchain`, `cartprod` and
/// `superpose` because three `tests/spec_coverage_measures.rs` tests reached
/// design-PR #73's no-laundering rider through `joint()` — the only measure
/// reachable from source whose mass is genuinely `%deferred`. That coverage now
/// lives in [`no_laundering_tests`], asserted directly against `product_mass` and
/// `lawof_type`, which is where it belonged: the old fixture could only reach
/// `Mass::Deferred` through one spellable expression, so `joint`'s arity and the
/// rider's coverage were coupled for no reason. With the rider covered, §04's rule
/// applies to all four. `vector` is the one head whose row
/// admits zero arguments, by the decision recorded on its catalogue row (`[]`
/// lowers to `(vector)`), and it never reaches here.
fn special_arity_check(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    let refuse = |inf: &mut Inferencer<'_, '_>, what: String, cite: &str| {
        inf.diags.push(crate::Diagnostic::error_at(
            id,
            format!("`{name}` {what} (spec §04 \"Calling conventions\": {cite})"),
        ));
        Some(Type::Failed(format!("{name}: {what}").into()))
    };
    // The measure combinators take their non-variadic inputs positionally (§04
    // "Calling conventions": "a keyword spelling such as `normalize(M = mu)` is a
    // static error"), and `cat` and `get` take unnamed variadic inputs. The count
    // is the catalogue row's business, so only the spelling is checked here.
    if let Some((distinguished, keywords_after, cite)) = positional_only_inputs(name)
        && !named.is_empty()
        && (args.len() < distinguished || !keywords_after)
    {
        return refuse(
            inf,
            format!(
                "takes its {} positionally and cannot be given {} keyword argument(s). Where \
                 the spec names such an input, the name identifies it in prose only — a call \
                 binds it by position. Pass the arguments positionally",
                if distinguished == 0 {
                    "inputs".to_string()
                } else {
                    format!("{distinguished} distinguished input(s)")
                },
                named.len()
            ),
            cite,
        );
    }
    let Some((declared, cite)) = special_arity(name) else {
        // §04: "Nullary calls (`f()`) are not allowed." Only for a head with NO
        // declared arity anywhere: [`arity_check`] returns `Some` on a mismatch
        // and `None` on a fitting call, so a row that ADMITS zero arguments has
        // already ruled and must not be second-guessed here.
        //
        // `vector` is that row, and the case is real: §04 "Multi-axis
        // aggregation" says of `output_axes` that "The empty axis list `[]` is
        // legal and denotes full reduction to a scalar", and an empty axis list
        // lowers to `(vector)`. The full-contraction spelling `g: s[] := …` in
        // `flatppl-examples/examples/dminus-to-3pi-amplitude.flatppl` writes
        // seven of them.
        if args.is_empty()
            && named.is_empty()
            && crate::catalogue::builtin().base_arity(name).is_none()
        {
            return refuse(
                inf,
                "is called with no arguments".to_string(),
                "\"Nullary calls (`f()`) are not allowed.\"",
            );
        }
        return None;
    };
    let (pos, kw) = (args.len(), named.len());
    let total = pos + kw;
    match declared {
        SpecialArity::DistinguishedPlusTail(n) => {
            // The distinguished prefix is positional; the tail is the callback's
            // additional arguments, checked when they bind.
            if pos < n {
                return refuse(
                    inf,
                    format!(
                        "takes {n} distinguished input(s) positionally, which cannot \
                         be passed by keyword; got {pos} positional argument(s). Where \
                         the spec names such an input, the name identifies it in prose \
                         only — a call binds it by position"
                    ),
                    cite,
                );
            }
        }
        SpecialArity::Distinguished(n) => {
            // Spelling first: on a keyword call the COUNT may well be right, and
            // reporting "takes 3 inputs, got 3" would be both true and useless.
            if kw > 0 {
                return refuse(
                    inf,
                    format!(
                        "takes {n} distinguished input(s), which cannot be passed by \
                         keyword; got {kw} keyword argument(s). Where the spec names \
                         such an input, the name identifies it in prose only — a call \
                         binds it by position. Pass the arguments positionally"
                    ),
                    cite,
                );
            }
            if pos != n {
                return refuse(inf, format!("takes {n} input(s), got {pos}"), cite);
            }
        }
        SpecialArity::DistinguishedPlusNamed(n) => {
            if pos != n {
                return refuse(
                    inf,
                    format!("takes {n} distinguished input(s), got {pos} positional argument(s)"),
                    cite,
                );
            }
        }
        SpecialArity::VariadicPositional(n) => {
            if kw > 0 {
                return refuse(
                    inf,
                    format!("takes unnamed inputs, got {kw} keyword argument(s)"),
                    cite,
                );
            }
            if pos < n {
                return refuse(inf, format!("takes at least {n} input(s), got {pos}"), cite);
            }
        }
        SpecialArity::DistinguishedPlusEither(n) => {
            // The distinguished prefix must be positional; the variadic tail may
            // be spelled either way, so only the prefix is checked here.
            if pos < n {
                return refuse(
                    inf,
                    format!(
                        "takes {n} distinguished input(s) positionally, which cannot \
                         be passed by keyword; got {pos} positional argument(s). Where \
                         the spec names such an input, the name identifies it in prose \
                         only — a call binds it by position"
                    ),
                    cite,
                );
            }
            if total < n + 1 {
                return refuse(
                    inf,
                    format!("takes at least {} input(s), got {total}", n + 1),
                    cite,
                );
            }
        }
        SpecialArity::VariadicEither(n) => {
            if total < n {
                return refuse(
                    inf,
                    format!("takes at least {n} input(s), got {total}"),
                    cite,
                );
            }
        }
        SpecialArity::VariadicNamedOrOneAggregate(n) => {
            // The positional spelling is the §03 record–table conversion, which
            // takes exactly one argument; anything else must be named.
            if kw == 0 && pos == 1 {
                return None;
            }
            if pos > 0 && kw > 0 {
                return refuse(
                    inf,
                    format!(
                        "takes named inputs, or the one positional argument of the \
                         §03 record-table conversion; got {pos} positional and {kw} \
                         keyword argument(s)"
                    ),
                    cite,
                );
            }
            if total < n {
                return refuse(
                    inf,
                    format!("takes at least {n} input(s), got {total}"),
                    cite,
                );
            }
        }
    }
    None
}

/// Reject a call whose argument falls outside the domain its row documents.
/// `None` when every argument is admissible, or when no argument's type is
/// concrete enough to decide — a `%deferred`, `%any` or type-variable argument
/// is never refused, so partial inference stays honest and only a definite
/// mismatch is a static error.
///
/// Three domains are checked, each anchored on the spec sentence that states it:
///
/// - the boolean-only domain of the conditional and logical heads
///   ([`refuse_nonboolean_domain`]),
/// - a string where the row documents no textual parameter
///   ([`refuse_string_argument`]),
/// - a non-measure where a §06 operation documents a measure
///   ([`refuse_nonmeasure_operand`]).
///
/// Ordered so the most specific refusal reports first; each returns at most one
/// diagnostic set, because a second opinion on the same call adds noise.
///
/// The scalar-KIND tags in a row's `params` are deliberately not swept
/// generically here. `ParamSig` documents the spec's declared domain, but §03
/// "Scalar value categories and sets" admits the canonical inclusions
/// (`booleans` $\subset$ `integers` $\subset$ `reals`) "where specified by the
/// language", and §07's Domains column does not say per row whether it means the
/// declared set or that set's embedding closure. Sweeping the tags would refuse
/// `Bernoulli(p = true)` and `eye(3.0)`, neither of which any sentence forbids.
/// The three checks below each rest on a sentence that names its exclusion
/// outright.
fn domain_check(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    refuse_operand_outside_operator_domain(inf, name, args, named)
        .or_else(|| refuse_nonboolean_domain(inf, name, args))
        .or_else(|| refuse_string_argument(inf, name, args))
        .or_else(|| refuse_nonmeasure_operand(inf, name, args))
        .or_else(|| refuse_bad_rand_operands(inf, name, args))
        .or_else(|| refuse_bad_rng_seed(inf, name, args))
        .or_else(|| refuse_constant_outside_declared_set(inf, name, args))
        .or_else(|| crate::boundary::refuse_bad_likelihood_point(inf, name, args))
        .or_else(|| crate::boundary::refuse_bad_chain_boundary(inf, name, args, named))
}

/// The §07 "Operator-equivalent functions" rows and the Domains cell each states, as
/// the spec writes it. The cell is quoted in the diagnostic.
fn operator_domain(name: &str) -> Option<&'static str> {
    Some(match name {
        "add" | "sub" => "scalars or arrays of same shape (real or complex)",
        "mul" => {
            "scalars, matrix-matrix, matrix-vector, scalar-matrix, matrix-scalar, scalar-vector, \
             vector-scalar, transposed-vector–vector, vector–transposed-vector, \
             transposed-vector–matrix"
        }
        "divide" => "scalars, array-scalar, transposed-vector–scalar (real or complex)",
        "neg" => "scalars or arrays (real or complex)",
        "pow" => "scalars (real or complex; complex extension via principal branch)",
        "equal" | "unequal" => "`integers`, `booleans`, strings",
        "lt" | "le" | "gt" | "ge" => "`reals`",
        _ => return None,
    })
}

/// What an operand is, when no §07 operator-equivalent row admits it: a record, tuple,
/// table, measure, kernel, function, likelihood, module or RNG state, or an array of
/// such. An ordering comparison also excludes a complex scalar, since `reals` does not
/// contain `complexes`. `None` admits the operand, including every unresolved type
/// (`%deferred`, `%any`, a type variable): only a definite mismatch is an error.
///
/// Booleans and integers stay admitted: §03 embeds `booleans` in `integers` and
/// `integers` in `reals` "where specified by the language", and the arithmetic rows say
/// "real or complex". This is the KIND of the operand, not the scalar-embedding question
/// that [`domain_check`] declines to sweep.
fn operand_outside_operator_domain(name: &str, ty: &Type) -> Option<&'static str> {
    match ty {
        Type::Record(_) => Some("a record"),
        Type::Tuple(_) => Some("a tuple"),
        Type::Table { .. } => Some("a table"),
        Type::Measure { .. } => Some("a measure"),
        Type::Kernel { .. } => Some("a kernel"),
        Type::Function { .. } => Some("a function"),
        Type::Likelihood { .. } => Some("a likelihood"),
        Type::Module => Some("a module"),
        Type::RngState => Some("an RNG state"),
        Type::Scalar(ScalarType::Complex) if matches!(name, "lt" | "le" | "gt" | "ge") => {
            Some("a complex number")
        }
        Type::Scalar(ScalarType::Real | ScalarType::Complex)
            if matches!(name, "equal" | "unequal") =>
        {
            Some("a non-discrete number")
        }
        // §07 gives each arithmetic row real or complex arrays, and a vector of vectors
        // is not a matrix (§03).
        Type::Array { elem, .. } | Type::TVector { elem, .. }
            if matches!(**elem, Type::Array { .. } | Type::TVector { .. })
                && matches!(name, "add" | "sub" | "mul" | "divide" | "pow" | "neg") =>
        {
            Some("an array of arrays")
        }
        Type::Array { elem, .. } | Type::TVector { elem, .. } => {
            operand_outside_operator_domain(name, elem).map(|_| "an array of non-numbers")
        }
        _ => None,
    }
}

/// How an operand of an arithmetic row sits in the row's shape clauses.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// A scalar.
    Scalar,
    /// A one-dimensional array of scalars.
    Vector,
    /// A flat rank-2 array of scalars.
    Matrix,
    /// A transposed vector.
    TVector,
    /// An array of three or more dimensions.
    Higher,
    /// Anything the shape clauses do not classify: unresolved types and nested arrays
    /// (a vector of vectors is not a matrix, §03).
    Unclassified,
}

fn shape_of(ty: &Type) -> Shape {
    match ty {
        Type::Scalar(_) => Shape::Scalar,
        Type::TVector { elem, .. } if matches!(**elem, Type::Scalar(_)) => Shape::TVector,
        Type::Array { shape, elem } if matches!(**elem, Type::Scalar(_)) => match shape.len() {
            0 => Shape::Unclassified,
            1 => Shape::Vector,
            2 => Shape::Matrix,
            _ => Shape::Higher,
        },
        _ => Shape::Unclassified,
    }
}

/// The shape clauses of the arithmetic rows of §07 "Operator-equivalent functions":
/// `add`/`sub` take scalars or arrays of the same shape, `divide` scalars, an array
/// over a scalar, or a transposed vector over a scalar, `pow` scalars only, and `mul`
/// exactly the listed pairs (scalars, matrix-matrix, matrix-vector, scalar-matrix,
/// scalar-vector, transposed-vector–vector, vector–transposed-vector,
/// transposed-vector–matrix). Only a definite mismatch is refused: an operand whose
/// shape is not classified is never an accusation.
fn operand_shapes_outside_domain(name: &str, a: &Type, b: &Type) -> bool {
    use Shape::*;
    let (sa, sb) = (shape_of(a), shape_of(b));
    if sa == Unclassified || sb == Unclassified {
        return false;
    }
    let collection = |s: Shape| !matches!(s, Scalar | Unclassified);
    let same_extent = |x: &Type, y: &Type| match (x, y) {
        (Type::Array { shape: p, .. }, Type::Array { shape: q, .. }) => {
            p.len() == q.len()
                && p.iter()
                    .zip(q.iter())
                    .all(|(m, n)| !matches!((m, n), (Dim::Static(m), Dim::Static(n)) if m != n))
        }
        (Type::TVector { len: p, .. }, Type::TVector { len: q, .. }) => {
            !matches!((p, q), (Dim::Static(m), Dim::Static(n)) if m != n)
        }
        _ => false,
    };
    match name {
        "add" | "sub" => match (collection(sa), collection(sb)) {
            (false, false) => false,
            (true, true) => !same_extent(a, b),
            _ => true,
        },
        // The divisor is the discriminator: a scalar (§05 "`/` requires a scalar divisor").
        "divide" => collection(sb),
        "pow" => collection(sa) || collection(sb),
        "mul" => {
            let listed = matches!(
                (sa, sb),
                (Scalar, Scalar)
                    | (Scalar, Vector)
                    | (Vector, Scalar)
                    | (Scalar, Matrix)
                    | (Matrix, Scalar)
                    | (Matrix, Matrix)
                    | (Matrix, Vector)
                    | (TVector, Vector)
                    | (Vector, TVector)
                    | (TVector, Matrix)
            );
            // The contracted extents must agree where both are static: the columns of the
            // left matrix, or the length of the transposed vector, against the leading
            // extent of the right operand.
            let static_extent = |t: &Type, axis: usize| match t {
                Type::TVector {
                    len: Dim::Static(n),
                    ..
                } if axis == 0 => Some(*n),
                Type::Array { shape, .. } => match shape.get(axis) {
                    Some(Dim::Static(n)) => Some(*n),
                    _ => None,
                },
                _ => None,
            };
            let contracted = match (sa, sb) {
                (Matrix, Matrix) | (Matrix, Vector) => (static_extent(a, 1), static_extent(b, 0)),
                (TVector, Vector) | (TVector, Matrix) => (static_extent(a, 0), static_extent(b, 0)),
                _ => (None, None),
            };
            !listed || matches!(contracted, (Some(m), Some(n)) if m != n)
        }
        _ => false,
    }
}

/// Refuse an operand outside the stated domain of a §07 "Operator-equivalent functions"
/// row (`add`, `sub`, `mul`, `divide`, `neg`, `pow`, `equal`, `unequal`, `lt`, `le`,
/// `gt`, `ge`) with an error located at the operand.
///
/// A record, tuple or table is in no row's domain. Without this check `add(r, 1.5)`
/// typed `%deferred`, because the per-row arms promote operand types without asking
/// what they are, so `record(a = 1, b = 2) + 10.0` passed inference and a later stage
/// met an expression with no value. A sole positional record or table splats under §04
/// "Calling conventions" and so is not an operand itself.
fn refuse_operand_outside_operator_domain(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    let domain = operator_domain(name)?;
    // A comparison over an array, transposed vector or table keeps its own, more
    // specific refusal (`refuse_nonscalar_operand`), which names the elementwise route.
    if matches!(name, "equal" | "unequal" | "lt" | "le" | "gt" | "ge")
        && nonscalar_operand(args, named).is_some()
    {
        return None;
    }
    if let ([(a, ta, _), (_, tb, _)], true) = (args, named.is_empty())
        && operand_outside_operator_domain(name, ta).is_none()
        && operand_outside_operator_domain(name, tb).is_none()
        && operand_shapes_outside_domain(name, ta, tb)
    {
        // §05 "No implicit operator broadcasting": the elementwise spelling is the dotted
        // operator, which applies §04 broadcasting.
        let dotted = match name {
            "add" => "`a .+ b`",
            "sub" => "`a .- b`",
            "mul" => "`a .* b`",
            "divide" => "`a ./ b`",
            _ => "`a .^ b`",
        };
        inf.diags.push(crate::Diagnostic::error_at(
            *a,
            format!(
                "`{name}` operands are outside its domain: spec §07 \"Operator-equivalent \
                 functions\" gives `{name}` the domain \"{domain}\". For the elementwise \
                 operation write {dotted}"
            ),
        ));
        // `mul_type` names a contracted-extent mismatch more precisely than the row does.
        let failed = match (name, mul_type(args)) {
            ("mul", failed @ Type::Failed(_)) => failed,
            _ => Type::Failed(format!("`{name}` applied to operands outside its domain").into()),
        };
        return Some(failed);
    }
    let splatting = named.is_empty() && args.len() == 1;
    let operands = args
        .iter()
        .map(|(node, ty, _)| (*node, ty))
        .chain(named.iter().map(|(_, node, ty, _)| (*node, ty)));
    // The sole positional record or table splats when its names are the row's own
    // parameters (`add(record(a = 1, b = 2))`); any other is itself the operand.
    let parameters = crate::catalogue::builtin().base_param_names(name);
    let splats = |ty: &Type, inf: &Inferencer<'_, '_>| {
        let fields: Vec<&str> = match ty {
            Type::Record(fields) => fields.iter().map(|(n, _)| inf.module.resolve(*n)).collect(),
            Type::Table { columns, .. } => columns
                .iter()
                .map(|(n, _)| inf.module.resolve(*n))
                .collect(),
            _ => return false,
        };
        parameters.is_some_and(|names| fields.iter().all(|f| names.iter().any(|n| n == f)))
    };
    for (node, ty) in operands {
        if splatting && splats(ty, inf) {
            continue;
        }
        let what = match operand_outside_operator_domain(name, ty) {
            Some(what) => what,
            // A set is not a value of any scalar or array type (§03 "Sets"), so no row
            // admits it; it carries no inferred type, so it is read structurally.
            None if !matches!(
                ty,
                Type::Scalar(_) | Type::Array { .. } | Type::TVector { .. }
            ) && set_expr_valueset(inf, Some(node)) != ValueSet::Unknown =>
            {
                "a set"
            }
            None => continue,
        };
        // §07: exact equality "is restricted to discrete domains"; its sentence names the
        // functions that give a discrete result, and `iszero` for an exact test of zero.
        let way_out = match what {
            "a non-discrete number" => {
                ". Compare a discrete result instead, such as `integer(x)`, `floor(x)`, \
                 `ceil(x)` or `round(x)`, or test a zero with `iszero(x)`"
            }
            "an array of arrays" => {
                ". `rowstack(…)` makes a matrix of a vector of vectors (spec §03), and a dotted \
                 operator applies the row to each element (spec §04)"
            }
            _ => "",
        };
        inf.diags.push(crate::Diagnostic::error_at(
            node,
            format!(
                "`{name}` operand is {what}, outside its domain: spec §07 \"Operator-equivalent \
                 functions\" gives `{name}` the domain \"{domain}\"{way_out}"
            ),
        ));
        return Some(Type::Failed(
            format!("`{name}` applied to an operand outside its domain").into(),
        ));
    }
    None
}

/// `rand(m, rstate)` accepts one closed measure and one explicit RNG state.
fn refuse_bad_rand_operands(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
) -> Option<Type> {
    if name != "rand" {
        return None;
    }
    for (index, (node, ty, _)) in args.iter().enumerate() {
        let valid = match ty {
            Type::Deferred | Type::Failed(_) | Type::Any | Type::Var(_) => true,
            Type::Measure { .. } => index == 0,
            Type::Kernel { inputs, .. } => index == 0 && inputs.is_empty(),
            Type::RngState => index == 1,
            _ => false,
        };
        if !valid {
            let required = if index == 0 {
                "a closed measure"
            } else {
                "an RNG state"
            };
            inf.diags.push(crate::Diagnostic::error_at(
                *node,
                format!(
                    "`rand(m, rstate)` argument {} must be {required} (spec §07)",
                    index + 1
                ),
            ));
            return Some(Type::Failed(
                format!("rand: argument {} must be {required}", index + 1).into(),
            ));
        }
    }
    None
}

/// 2^64 — the exclusive upper bound on `rnginit`'s integer seed. An `i64`
/// literal never reaches it, so the bound documents the spec's set rather than
/// gating a case the front end can produce; see [`refuse_bad_rng_seed`].
const RNG_SEED_BOUND: i128 = 1 << 64;

/// Reject an `rnginit` seed that is neither a byte vector nor an integer.
///
/// §07 `rnginit` states two forms: "`rngseed` must be a seed vector of bytes
/// (integers in $\{0, \ldots, 255\}$)", and "A single integer
/// $n \in \{0, \ldots, 2^{64} - 1\}$ is also accepted, and denotes the byte
/// vector of the 8-byte little-endian unsigned encoding of $n$."
///
/// Unenforced, both `rnginit(1.5)` and `rnginit(eye(3))` typed `rngstates` at
/// exit 0 — the `"rnginit"` arm answers `Type::RngState` whatever the argument
/// is, and the row's `Vector(Integer)` parameter tag is documentation that
/// nothing sweeps ([`domain_check`]'s doc comment says why the tags are not
/// swept generically). This is the narrow check the two stated forms support.
///
/// A boolean seed is refused with the rest: §03 promotes `true` to one "in
/// arithmetic contexts", and a seed is not an arithmetic context — §07 says a
/// single INTEGER, and `rnginit(true)` is far likelier a mistake than a request
/// for seed 1.
///
/// The RANGE is only checked for a written constant, and only downward: a
/// negative seed is in neither form's set. The upper bound cannot fire from
/// source, because a literal above `i64::MAX` is already a lex error
/// ("invalid integer literal"), which leaves $[2^{63}, 2^{64})$ unwritable as a
/// literal here — a front-end limit, not an `rnginit` rule. The byte vector's
/// own $\{0, \ldots, 255\}$ element range is checked for a written vector, by
/// the same rule: an element outside it is in neither form's set.
fn refuse_bad_rng_seed(inf: &mut Inferencer<'_, '_>, name: &str, args: &[ArgInfo]) -> Option<Type> {
    if name != "rnginit" {
        return None;
    }
    let (node, ty, _) = args.first()?;
    let form = match ty {
        Type::Scalar(ScalarType::Integer) => "an integer seed",
        Type::Array { shape, elem } if shape.len() == 1 => {
            if matches!(**elem, Type::Scalar(ScalarType::Integer)) {
                "a byte vector"
            } else {
                ""
            }
        }
        // Not decided against, so not refused — same abstention as every other
        // check in `domain_check`.
        Type::Deferred | Type::Failed(_) | Type::Any | Type::Var(_) => return None,
        _ => "",
    };
    if form.is_empty() {
        inf.diags.push(crate::Diagnostic::error_at(
            *node,
            format!(
                "`rnginit` seed must be a byte vector (integers in 0..255) or a single \
                 integer in 0..2^64-1, got {}: spec §07 `rnginit` states those two \
                 forms and no other",
                describe_seed(ty),
            ),
        ));
        return Some(Type::Failed("rnginit: seed is neither form".into()));
    }
    if form == "a byte vector" {
        // A written byte vector is checked element by element; any other vector
        // is a run-time value this pass cannot read.
        let literal = resolve_binding_refs(inf, *node);
        if let Node::Call(call) = inf.module.node(literal)
            && matches!(call.head, CallHead::Builtin(op) if inf.module.resolve(op) == "vector")
        {
            let elements = call.args.to_vec();
            for element in elements {
                if let Some(byte) = written_int(inf, element)
                    && !(0..=255).contains(&byte)
                {
                    inf.diags.push(crate::Diagnostic::error_at(
                        element,
                        format!(
                            "`rnginit` byte-vector seed has element {byte}, outside 0..255: spec \
                             §07 `rnginit` states that a byte-vector seed contains integers in \
                             $\\{{0, \\ldots, 255\\}}$"
                        ),
                    ));
                    return Some(Type::Failed("rnginit: seed byte out of range".into()));
                }
            }
        }
    }
    if form == "an integer seed" {
        let n = written_int(inf, *node)?;
        if n < 0 || i128::from(n) >= RNG_SEED_BOUND {
            inf.diags.push(crate::Diagnostic::error_at(
                *node,
                format!(
                    "`rnginit` integer seed is {n}, outside 0..2^64-1: spec §07 \
                     `rnginit` accepts \"a single integer $n \\in \\{{0, \\ldots, \
                     2^{{64}} - 1\\}}$\""
                ),
            ));
            return Some(Type::Failed("rnginit: seed out of range".into()));
        }
    }
    None
}

/// How a refused `rnginit` seed reads in its diagnostic. An array names its
/// ELEMENT as well as its rank: `[1.5, 2.5]` fails on the element and
/// `eye(3)` on the rank, and "a rank-1 array" alone would not say which.
fn describe_seed(ty: &Type) -> String {
    match ty {
        Type::Scalar(k) => format!("the {k} scalar"),
        Type::Array { shape, elem } => match &**elem {
            Type::Scalar(k) => format!("a rank-{} array of {k}", shape.len()),
            _ => format!("a rank-{} array of a non-scalar element", shape.len()),
        },
        Type::TVector { .. } => "a transposed vector".to_string(),
        Type::Record(fields) => format!("a {}-field record", fields.len()),
        Type::Table { columns, .. } => format!("a {}-column table", columns.len()),
        Type::Tuple(parts) => format!("a {}-component tuple", parts.len()),
        Type::RngState => "an RNG state".to_string(),
        Type::Module => "a module reference".to_string(),
        Type::Measure { .. } => "a measure".to_string(),
        Type::Kernel { .. } => "a kernel".to_string(),
        Type::Function { .. } => "a function".to_string(),
        Type::Likelihood { .. } => "a likelihood".to_string(),
        // Never reached: `refuse_bad_rng_seed` abstains on an undecided type
        // before describing it.
        Type::Deferred | Type::Failed(_) | Type::Any | Type::Var(_) => {
            "an undecided type".to_string()
        }
    }
}

/// The value of a scalar constant WRITTEN at this node, as an `f64`, or `None`
/// when the argument is not a statically-known constant.
///
/// Deliberately separate from [`crate::consteval`], which folds INTEGERS only —
/// shapes are the only fixed values it needs, so it answers `Dynamic` for a real
/// or a boolean, which are exactly the constants a parameter value set has to be
/// tested against. This walk is the narrower thing the check needs and nothing
/// more: a literal, a `neg` over one (how the lexer spells `-3`), a predefined
/// numeric constant, or a self-module reference to a binding that is one.
///
/// A boolean reads as `0` or `1`, per §03 "Bool": "In arithmetic contexts, `false`
/// is promoted to zero and `true` to one." That promotion is what makes
/// `Normal(sigma = false)` decidable — `false` IS the value `0`, and `0` is not
/// in `posreals`.
///
/// `pi` and `inf` are included because §03 "Predefined constants" gives both a
/// value, and §03's infinity note makes `inf` a legal member of `posreals`,
/// `nonnegreals` and `reals` — so admitting it is a real answer, not an
/// abstention. `im` is absent: a complex constant is not a point on the real line
/// and no `ParamSet` ranges over one.
///
/// The alias walk is bounded for the same reason [`written_string`]'s is: a
/// self-referential binding is a well-formed parse whose cycle a later pass
/// reports, and this runs before that.
fn written_scalar(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<f64> {
    /// Alias hops to follow before giving up. Any real chain is far shorter.
    const MAX_ALIAS_HOPS: usize = 64;
    let mut node = node;
    let mut negate = false;
    for _ in 0..MAX_ALIAS_HOPS {
        let v = match inf.module.node(node) {
            Node::Lit(Scalar::Int(n)) => Some(*n as f64),
            Node::Lit(Scalar::Real(x)) => Some(*x),
            Node::Lit(Scalar::Bool(b)) => Some(f64::from(*b)),
            Node::Const(c) => match inf.module.resolve(*c) {
                "pi" => Some(std::f64::consts::PI),
                "inf" => Some(f64::INFINITY),
                _ => return None,
            },
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
                continue;
            }
            Node::Call(c) => {
                let CallHead::Builtin(op) = c.head else {
                    return None;
                };
                if inf.module.resolve(op) != "neg" {
                    return None;
                }
                negate = !negate;
                node = *c.args.first()?;
                continue;
            }
            _ => return None,
        };
        return v.map(|v| if negate { -v } else { v });
    }
    None
}

/// Reject a constant argument whose VALUE lies outside the value set the spec
/// declares for that parameter.
///
/// §08 states each distribution parameter's set outright — `Normal`'s is
/// "`mu = elementof(reals)`" and "`sigma = elementof(posreals)`" — and §04
/// "Calling conventions" is what binds an argument to that parameter, by position
/// or by name. §03 "Sets" fixes what each named set contains, including the
/// bounds: `posreals` is $(0, +\infty]$, so it excludes `0`.
///
/// Unenforced, `Normal(mu = "bad", sigma = false)` typed a normalized `%measure`
/// over `reals` at exit 0. The string half is [`refuse_string_argument`]'s; this
/// is the other half — `false` IS `0` (§03 "Bool"), and a normal distribution
/// with zero standard deviation is not a measure §08 defines. The determiniser
/// then lowered a density dividing by that zero.
///
/// **Constants only, and that is the whole scope of the rule.** A computed
/// argument is not tested: no engine can decide `sigma = f(data)` statically, and
/// §03 says a deterministically computed node's value set is only "a conservative
/// superset of the values that `x` can take". So the check answers on what it can
/// see — a literal, a negated literal, a predefined constant, or a binding holding
/// one — and stays silent otherwise. That is why `Bernoulli(p = true)` still
/// types: `1` is in the CLOSED `unitinterval` $[0, 1]$, so the check runs and
/// ADMITS it, rather than declining to look.
///
/// A row lists a set only where the spec states one, so most positions are
/// `ParamSet::Unconstrained` and never reach a comparison. See [`ParamSet`] for
/// which parameters are deliberately left unconstrained — a set-valued argument
/// like `Uniform`'s `support`, and every vector or matrix parameter, whose §08
/// constraint is per-element rather than a range on one number.
fn refuse_constant_outside_declared_set(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
) -> Option<Type> {
    let cat = crate::catalogue::builtin();
    let sets: Vec<crate::catalogue::ParamSet> = cat.base_param_sets(name).to_vec();
    if sets.is_empty() {
        return None;
    }
    let names: Vec<String> = cat
        .base_param_names(name)
        .map(<[String]>::to_vec)
        .unwrap_or_default();
    let section = cat.base_param_section(name);
    let mut refused = None;
    for (i, (node, _, _)) in args.iter().enumerate() {
        let Some(set) = sets.get(i) else { break };
        let Some(value) = written_scalar(inf, *node) else {
            continue;
        };
        // `None` is "this parameter states no set", not "outside it".
        if set.contains(value) != Some(false) {
            continue;
        }
        let which = names
            .get(i)
            .map(|n| format!("`{n}`"))
            .unwrap_or_else(|| format!("argument {}", i + 1));
        inf.diags.push(crate::Diagnostic::error_at(
            *node,
            format!(
                "`{name}`'s {which} is written as `{value}`, which is outside its \
                 declared value set `{}`: spec {section} states the parameter's set, \
                 and spec §03 \"Sets\" fixes what that set contains. A computed \
                 argument is not checked — only a constant, whose value is known here",
                set.describe(),
            ),
        ));
        refused = Some(which);
    }
    refused.map(|which| Type::Failed(format!("{name}: {which} is outside its declared set").into()))
}

/// Reject an argument whose shape class is outside its parameter's declared domain:
/// a non-scalar where §08 or §09 declares `elementof(<scalar set>)` or §07 a scalar, a
/// non-vector where a row declares a vector, a non-matrix where it declares a matrix.
/// The classes come from the catalogue row ([`sig_param_shapes`]), so every §07, §08
/// and §09 row is checked by the same rule.
///
/// §07 "Elementary functions" states the scalar case outright, "All accept scalar
/// arguments and return scalar results", and §05 "No implicit operator broadcasting"
/// leaves `broadcast` and the dotted call as the elementwise route. Without this,
/// `Normal(mu = v, sigma = v)` typed a measure over `reals` and `linsolve(v, v)` a
/// vector, both at exit 0. §03 "Arrays" decides the matrix case for a nested array:
/// "Vectors of vectors are not interpreted as matrices implicitly".
///
/// `cell` marks the per-cell call of a `broadcast`, whose arguments are cells: the
/// broadcast is already the elementwise route, so no dotted remedy is offered.
///
/// An unresolved argument (`%deferred`, `%any`, a type variable, `%failed`) is never
/// refused: only a decided mismatch is an error.
///
/// [`sig_param_shapes`]: crate::catalogue::sig_param_shapes
fn refuse_argument_outside_declared_shape(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    cell: bool,
) -> Option<Type> {
    // `refuse_nonscalar_operand` owns the connectives' array operands and names their
    // dotted route; the catalogue marks no such class.
    if matches!(name, "land" | "lor" | "lxor" | "lnot") {
        return None;
    }
    let cat = crate::catalogue::builtin();
    let sig = cat.base(name)?;
    let names = cat.base_param_names(name).unwrap_or_default();
    let who = format!("`{name}`");
    refuse_shape_outside_sig(
        inf,
        &who,
        name,
        cat.base_param_section(name),
        sig,
        names,
        args,
        cell,
    )
}

/// [`refuse_argument_outside_declared_shape`] for any catalogue row: `who` names the
/// callee in the message, `head` is the name a dotted remedy calls.
#[allow(clippy::too_many_arguments)]
fn refuse_shape_outside_sig(
    inf: &mut Inferencer<'_, '_>,
    who: &str,
    head: &str,
    section: &str,
    sig: &crate::catalogue::Sig,
    names: &[String],
    args: &[ArgInfo],
    cell: bool,
) -> Option<Type> {
    let shapes = crate::catalogue::sig_param_shapes(sig, args.len());
    // `div` and `mod` own an integer check that also reaches broadcast cells and
    // names `divide` for real division (`integer_arithmetic_type`).
    let integral = if matches!(head, "div" | "mod") {
        Vec::new()
    } else {
        crate::catalogue::sig_param_integral(sig, args.len())
    };
    let distribution = matches!(sig, crate::catalogue::Sig::Distribution { .. });
    let mut refused = None;
    for (i, (node, ty, _)) in args.iter().enumerate() {
        let which = names
            .get(i)
            .map(|n| format!("`{n}`"))
            .unwrap_or_else(|| format!("argument {}", i + 1));
        // §03 "Integer" and "Bool": an integer or boolean domain is checked by the
        // argument's TYPE. `eye(2.0)` passes a real, whatever number it holds.
        // A boolean is an integer, but an integer is not a boolean (§03 "booleans ⊂
        // integers ⊂ reals"; §03 "Bool": zero and one are not converted implicitly).
        if let Some(Some(want)) = integral.get(i)
            && let Type::Scalar(got) = ty
            && match want {
                crate::catalogue::ScalarTag::Boolean => *got != ScalarType::Boolean,
                _ => matches!(got, ScalarType::Real | ScalarType::Complex),
            }
        {
            let subject = if cell {
                format!("a `broadcast` cell of {who} passes {which}")
            } else {
                format!("{who}'s {which} is")
            };
            let message = non_integral_message(&subject, *got, section, *want);
            inf.diags.push(crate::Diagnostic::error_at(*node, message));
            refused = Some(which);
            continue;
        }
        let Some(Some(shape)) = shapes.get(i) else {
            continue;
        };
        // §08 declares its vector parameters "array of reals", and §03 "Arrays"
        // makes a transposed vector a distinct type.
        if distribution && *shape == ParamShape::Vector && matches!(ty, Type::TVector { .. }) {
            inf.diags.push(crate::Diagnostic::error_at(
                *node,
                format!(
                    "{who}'s {which} is a transposed vector, but spec {section} declares it an \
                     array; spec §03 \"Arrays\" makes transposed vectors a distinct type, so \
                     pass the vector itself, not `transpose(…)` of it"
                ),
            ));
            refused = Some(which);
            continue;
        }
        let Some(what) = shape_mismatch(*shape, ty) else {
            continue;
        };
        let remedy = match (shape, ty) {
            (ParamShape::Scalar, Type::Array { .. } | Type::TVector { .. }) if !cell => {
                format!(". Apply {who} elementwise with `{head}.(…)` (spec §04 \"Broadcasting\")")
            }
            (ParamShape::Matrix, _) if what == "a vector of vectors" => {
                ". Spec §03 \"Arrays\" does not read a vector of vectors as a matrix: build \
                 it with `rowstack(...)` or `colstack(...)`"
                    .to_string()
            }
            _ => String::new(),
        };
        let subject = if cell {
            format!("a `broadcast` cell of {who} passes {which}")
        } else {
            format!("{who}'s {which} is")
        };
        inf.diags.push(crate::Diagnostic::error_at(
            *node,
            format!(
                "{subject} {what}, but spec {section} declares it {}{remedy}",
                shape.describe()
            ),
        ));
        refused = Some(which);
    }
    refused.map(|which| {
        let who = who.trim_matches('`');
        Type::Failed(format!("{who}: {which} is outside its declared shape").into())
    })
}

/// Refuse `integer(x)` / `boolean(x)` of a WRITTEN literal that the conversion
/// cannot represent: §07 "Scalar restrictions and constructors" makes `integer`
/// return the integer equal to an integral `x` and `boolean` return `x` in
/// `booleans`, so `integer(2.5)` and `boolean(2)` are static errors. `-0.0` is 0. A
/// bound argument is not read here: only a literal's value is known.
fn refuse_lossy_literal_conversion(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
) -> Option<Type> {
    if !matches!(name, "integer" | "boolean") {
        return None;
    }
    let (node, _, _) = args.first()?;
    let value = written_literal(inf, *node)?;
    let representable = if name == "integer" {
        value.fract() == 0.0
    } else {
        value == 0.0 || value == 1.0
    };
    if representable {
        return None;
    }
    let want = if name == "integer" {
        "an integral value"
    } else {
        "0 or 1"
    };
    inf.diags.push(crate::Diagnostic::error_at(
        *node,
        format!(
            "`{name}` of `{value}`: spec §07 \"Scalar restrictions and constructors\" converts \
             only {want}, so this conversion has no result"
        ),
    ));
    Some(Type::Failed(
        format!("{name}: unrepresentable literal").into(),
    ))
}

/// The value of a numeric or boolean literal, possibly negated, as written. Unlike
/// [`written_scalar`] it does not follow a binding.
fn written_literal(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<f64> {
    match inf.module.node(node) {
        Node::Lit(Scalar::Int(n)) => Some(*n as f64),
        Node::Lit(Scalar::Real(x)) => Some(*x),
        Node::Lit(Scalar::Bool(b)) => Some(f64::from(*b)),
        Node::Call(c) => {
            let CallHead::Builtin(op) = c.head else {
                return None;
            };
            if inf.module.resolve(op) != "neg" || c.args.len() != 1 {
                return None;
            }
            written_literal(inf, c.args[0]).map(|x| -x)
        }
        _ => None,
    }
}

/// Refuse a real or complex index (or an array of them) in a `get` / `get0` selector.
/// An index is an integer by TYPE, alongside the separate 1-based rule: `v[1.0]` writes
/// a real, whatever number it holds.
fn refuse_real_index(inf: &mut Inferencer<'_, '_>, name: &str, args: &[ArgInfo]) -> bool {
    let mut refused = false;
    for (i, (node, ty, _)) in args.iter().enumerate().skip(1) {
        let elem = match ty {
            Type::Array { shape, elem } if shape.len() == 1 => elem.as_ref(),
            other => other,
        };
        let Type::Scalar(got @ (ScalarType::Real | ScalarType::Complex)) = elem else {
            continue;
        };
        let subject = format!(
            "`{name}`'s selector {i}{} is",
            if elem == ty { "" } else { " vector" }
        );
        let message = non_integral_message(
            &subject,
            *got,
            &format!("§07 `{name}`"),
            crate::catalogue::ScalarTag::Integer,
        );
        inf.diags.push(crate::Diagnostic::error_at(*node, message));
        refused = true;
    }
    refused
}

/// The refusal of a real or complex argument where an integer or boolean domain is
/// declared. `subject` ends in the verb, as in "`eye`'s `n` is".
fn non_integral_message(
    subject: &str,
    got: ScalarType,
    section: &str,
    want: crate::catalogue::ScalarTag,
) -> String {
    let (want, remedy) = match want {
        crate::catalogue::ScalarTag::Boolean => ("a boolean", "`true`, `false` or `boolean(…)`"),
        _ => (
            "an integer",
            "an integer literal (`2`, not `2.0`) or `integer(…)`",
        ),
    };
    let got = match got {
        ScalarType::Complex => "a complex",
        ScalarType::Integer => "an integer",
        ScalarType::Boolean => "a boolean",
        ScalarType::Real => "a real",
    };
    format!(
        "{subject} {got}, but spec {section} declares it {want}: a domain of integers or \
         booleans is checked by type, not by value, so write {remedy}"
    )
}

/// What `ty` is when it is outside `shape`, or `None` when it fits or is unresolved.
fn shape_mismatch(shape: ParamShape, ty: &Type) -> Option<String> {
    let unresolved = |t: &Type| {
        matches!(
            t,
            Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_)
        )
    };
    let numeric = |t: &Type| matches!(t, Type::Scalar(_)) || unresolved(t);
    let what = match ty {
        Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_) => return None,
        Type::Scalar(_) if shape == ParamShape::Scalar => return None,
        Type::Scalar(_) => "a scalar",
        Type::Array { shape: dims, elem } => match (dims.len(), numeric(elem)) {
            (1, true) if shape == ParamShape::Vector => return None,
            (2, true) if shape == ParamShape::Matrix => return None,
            (1, true) => "a vector",
            (2, true) => "a matrix",
            (n, true) => return Some(format!("an array of {n} dimensions")),
            (1, false) if is_array_like(elem) => "a vector of vectors",
            (_, false) => "an array of non-scalars",
        },
        // §03 "Arrays": "The term vector will represent both non-transposed vectors
        // (one-dimensional arrays) and transposed vectors".
        Type::TVector { elem, .. } if shape == ParamShape::Vector && numeric(elem) => return None,
        Type::TVector { .. } => "a transposed vector",
        Type::Record(_) => "a record",
        Type::Tuple(_) => "a tuple",
        Type::Table { .. } => "a table",
        Type::Measure { .. } => "a measure",
        Type::Kernel { .. } => "a kernel",
        Type::Function { .. } => "a function",
        Type::Likelihood { .. } => "a likelihood",
        Type::Module => "a module",
        Type::RngState => "an RNG state",
    };
    Some(what.to_string())
}

/// The first `%failed` in `ty`: itself, or one held in an element, field, component or
/// measure domain.
pub(crate) fn first_failed(ty: &Type) -> Option<Type> {
    match ty {
        Type::Failed(_) => Some(ty.clone()),
        Type::Array { elem, .. } | Type::TVector { elem, .. } => first_failed(elem),
        Type::Record(fields)
        | Type::Table {
            columns: fields, ..
        } => fields.iter().find_map(|(_, t)| first_failed(t)),
        Type::Tuple(parts) => parts.iter().find_map(first_failed),
        Type::Measure { domain, .. } => first_failed(domain),
        Type::Likelihood { obstype, .. } => first_failed(obstype),
        _ => None,
    }
}

/// The type a catalogue row's call takes while a shaped argument is unresolved: the
/// `%failed` of a failed argument, or `%deferred` for a `%deferred` one.
///
/// A row's nominal result presumes each argument has its declared shape. `sqrt` of a
/// `%deferred` argument typed a confirmed real scalar although that argument could be a
/// vector, which a later scalar gate then trusted. A §08 distribution keeps its
/// §08 variate domain for a `%deferred` argument, which the row fixes whatever its
/// parameters' values. A `%failed` argument never reaches a row: `call_rule` fails the
/// call first. `%any` and a type variable are a placeholder's honest "no constraint"
/// (the `_` of `fn(sqrt(_))`), whose body type is the nominal result.
fn unresolved_shaped_argument(sig: &crate::catalogue::Sig, args: &[ArgInfo]) -> Option<Type> {
    let shapes = crate::catalogue::sig_param_shapes(sig, args.len());
    args.iter()
        .zip(shapes)
        .filter(|(_, shape)| shape.is_some())
        .find_map(|((_, ty, _), _)| match ty {
            Type::Deferred | Type::Failed(_) => Some(ty.clone()),
            _ => None,
        })
}

/// The argument positions of `name` whose domain §03 "Bool" states as boolean,
/// or `None` for every other head. `ifelse` constrains only `cond`; the four
/// connectives constrain every operand.
///
/// §03 "Bool" is the decisive sentence and it names the five heads itself:
/// "Conditional and logical constructs (`ifelse`, `land`, `lor`, `lnot`, `lxor`)
/// strictly require boolean arguments; zero and one are not implicitly converted
/// to booleans." §07 "Logic and conditionals" gives the same domains in its
/// tables — `booleans` for the four connectives, "`cond`: `booleans`" for
/// `ifelse` — so the two agree and the §03 sentence supplies the "not implicitly
/// converted" half that decides an integer operand.
///
/// `lor` is included although §03's list omits it: §07's table gives it the
/// `booleans` domain beside its three siblings, and §03's own list names it in
/// the parenthetical's sibling roster at [`refuse_nonscalar_operand`]. The
/// omission is a typo in one list, not a licence.
fn boolean_domain_positions(name: &str) -> Option<&'static [usize]> {
    match name {
        "land" | "lor" | "lxor" => Some(&[0, 1]),
        "lnot" => Some(&[0]),
        "ifelse" => Some(&[0]),
        _ => None,
    }
}

/// Reject a non-boolean scalar where §03 "Bool" requires a boolean.
///
/// [`refuse_nonscalar_operand`] already refuses an ARRAY or table operand on the
/// four connectives, on the separate ground that §07 gives them a scalar domain.
/// This check is about the scalar KIND, so it fires on the case that guard admits:
/// `land(1, 2)` typed `%boolean` at exit 0, and `ifelse(3, 4, 5)` typed the common
/// type of its branches, both handing the determiniser a predicate the author
/// never wrote. §03's "zero and one are not implicitly converted to booleans"
/// forbids exactly this, so the integer that looks like a truth value is refused
/// rather than silently accepted.
///
/// An array argument to `ifelse`'s `cond` is refused here too — `ifelse` is not on
/// the non-scalar guard's list, and §07's table gives `cond` the scalar domain
/// `booleans` while spelling `a`/`b`'s domain as `anything`. The remedy names the
/// dotted form, which is where an elementwise select belongs.
fn refuse_nonboolean_domain(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
) -> Option<Type> {
    let positions = boolean_domain_positions(name)?;
    let mut offender = None;
    for i in positions {
        let Some((node, ty, _)) = args.get(*i) else {
            continue;
        };
        // Only a concrete non-boolean scalar, or an array of them, is a decided
        // mismatch. `%any` covers a hole and a string, neither of which this
        // check owns.
        let kind = match ty {
            Type::Scalar(k) if *k != ScalarType::Boolean => *k,
            Type::Array { elem, .. } | Type::TVector { elem, .. } => match **elem {
                Type::Scalar(k) if k != ScalarType::Boolean => k,
                _ => continue,
            },
            _ => continue,
        };
        let which = if name == "ifelse" {
            "`cond`".to_string()
        } else {
            format!("argument {}", i + 1)
        };
        inf.diags.push(crate::Diagnostic::error_at(
            *node,
            format!(
                "`{name}` requires a boolean {which}, got {kind}: spec §03 \"Bool\" \
                 states that the conditional and logical constructs \"strictly require \
                 boolean arguments; zero and one are not implicitly converted to \
                 booleans\", and spec §07 \"Logic and conditionals\" gives the same \
                 `booleans` domain. Compare against a value instead, as in \
                 `unequal({which_expr}, 0)`, to get the boolean the domain wants",
                which_expr = if name == "ifelse" { "cond" } else { "x" },
            ),
        ));
        offender = Some(*i);
    }
    offender.map(|i| Type::Failed(format!("{name} argument {} is not boolean", i + 1).into()))
}

/// The argument positions of `name` whose domain admits a STRING, or `None` when
/// the row documents no textual parameter at all.
///
/// A string is not a §03 value: §03 "Scalar types" enumerates Real, Integer, Bool
/// and Complex and nothing else, and §11 gives strings no value type — this
/// crate's [`literal_type`] answers `%any` for one, which is what let a string
/// reach every numeric rule unchallenged. So a string argument is admissible only
/// where a spec sentence documents that parameter as textual, and this table is
/// that set, read off the sentence in each case:
///
/// - `load_data` — §07 "Data loading" gives `source` as a path or URL.
/// - `load_module` — §04 "Module composition": "`source` may be a file path or a
///   URL".
/// - `standard_module` — §04 "Standard modules": `name` is an official module name
///   and `compat` "a compatibility version string".
/// - `get`, `get0` — §03 "Records": "`r.name1` (lowers to `get(r, "name1")`)", so
///   every SELECTOR position is textual. Position 0 is the container and is not.
/// - `equal`, `unequal` — §07 "Operator-equivalent functions" gives this pair the
///   domains "`integers`, `booleans`, strings", the only §07 row that names
///   strings.
/// - `vector`, `cat` — the field-name selector LIST is spelled as an array
///   literal throughout the spec (§06 `disintegrate(["obs"], …)`, §04
///   `relabel(x, ["x", "y", "z"])`, §07 `get(_, ["a", "c"])`), and an array
///   literal lowers to `vector`. Refusing a string element here would refuse the
///   spec's own examples. §03 "Arrays" bites on the MIXED array instead, where a
///   string element sits beside a numeric one.
///
/// `usize::MAX` in a list means "every position from the previous entry onward",
/// used for the two variadic rows.
fn string_domain_positions(name: &str) -> Option<&'static [usize]> {
    match name {
        "load_data" | "load_module" => Some(&[0]),
        "standard_module" => Some(&[0, 1]),
        "get" | "get0" => Some(&[1, usize::MAX]),
        "equal" | "unequal" => Some(&[0, 1]),
        "vector" | "cat" => Some(&[0, usize::MAX]),
        _ => None,
    }
}

/// True when `positions` admits index `i`, expanding a trailing `usize::MAX` into
/// "and every later position".
fn position_admitted(positions: &[usize], i: usize) -> bool {
    if positions.last() == Some(&usize::MAX) {
        return positions
            .iter()
            .rev()
            .nth(1)
            .is_some_and(|first_variadic| i >= *first_variadic);
    }
    positions.contains(&i)
}

/// Reject a string literal at a parameter whose row documents no textual domain.
///
/// Unenforced, `sqrt("text")` typed a non-negative real and
/// `Normal(mu = "bad", sigma = 1.0)` typed a normalized measure over `reals`, both
/// at exit 0, because [`literal_type`] answers `%any` for a string and every
/// numeric rule reads `%any` as "not yet known". The determiniser then lowered a
/// density whose parameter record held a string. See [`string_domain_positions`]
/// for the rows that do document a textual parameter and the sentence for each.
///
/// Resolves through a self-module reference, so a string bound to a name and then
/// passed is caught as well as an inline literal. Anything that is not a written
/// string literal is left alone: this check refuses what it can see, and says
/// nothing about a string that arrives from `load_data`.
fn refuse_string_argument(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
) -> Option<Type> {
    let admitted = string_domain_positions(name);
    let mut refused = false;
    for (i, (node, _, _)) in args.iter().enumerate() {
        if admitted.is_some_and(|p| position_admitted(p, i)) {
            continue;
        }
        let Some(text) = written_string(inf, *node) else {
            continue;
        };
        inf.diags.push(crate::Diagnostic::error_at(
            *node,
            format!(
                "`{name}` argument {} is the string `\"{text}\"`, which is not a \
                 FlatPPL value: spec §03 \"Scalar types\" gives the scalar types as \
                 real, integer, boolean and complex, and no spec entry documents a \
                 textual parameter for `{name}`. Strings denote paths, module names \
                 and field-name selectors only",
                i + 1,
            ),
        ));
        refused = true;
    }
    refused.then(|| Type::Failed(format!("{name}: string argument").into()))
}

/// The text of a string WRITTEN at this node: a literal, or a self-module
/// reference to a binding whose right-hand side is one. Mirrors [`written_int`],
/// and for the same reason — a string carries no `Type`, so the only way to see
/// one is to look at the node.
///
/// **The reference walk is BOUNDED, and it has to be.** A self-referential
/// binding is a well-formed parse that inference reports as a cycle
/// (`crates/infer/tests/named_set_expression.rs`
/// `::a_self_referential_set_binding_reports_a_cycle_not_a_crash` pins
/// `s = s`), and this function runs on every argument of every builtin call —
/// including one whose binding is in that cycle, and before the cycle is
/// reported. An unbounded `loop` here HUNG that test. The bound is not a
/// semantic limit: a legitimate alias chain to a string literal is a handful of
/// hops, so exhausting it means the chain is pathological, and answering `None`
/// then is the honest reading — "no written string visible here" — which leaves
/// the cycle to the pass that owns it.
fn written_string(inf: &Inferencer<'_, '_>, node: NodeId) -> Option<String> {
    /// Alias hops to follow before giving up. Any real chain is far shorter.
    const MAX_ALIAS_HOPS: usize = 64;
    let mut node = node;
    for _ in 0..MAX_ALIAS_HOPS {
        match inf.module.node(node) {
            Node::Lit(Scalar::Str(s)) => return Some(s.to_string()),
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            _ => return None,
        }
    }
    None
}

/// The argument position of `name` that §06 documents as a MEASURE, with the
/// operation's own sentence for the diagnostic. `None` for every other head.
///
/// Each entry quotes the §06 prose that names the argument a measure, so the
/// refusal cites the operation the author wrote rather than a general rule:
///
/// - `normalize` — "given a measure $M$ with finite total mass".
/// - `truncate` — "restricts the support of measure `M` to the set `S`".
/// - `weighted`, `logweighted` — "reweight `base`", the second argument.
/// - `bayesupdate` — "`prior` reweighted by likelihood `L`", the second argument.
fn measure_operand(name: &str) -> Option<(usize, &'static str)> {
    match name {
        "normalize" => Some((
            0,
            "`normalize(M)` is \"given a measure $M$ with finite total mass\"",
        )),
        "truncate" => Some((
            0,
            "`truncate(M, S)` \"restricts the support of measure `M` to the set `S`\"",
        )),
        "weighted" | "logweighted" => Some((1, "the second argument is the `base` measure")),
        "bayesupdate" => Some((1, "the second argument is the `prior` measure")),
        _ => None,
    }
}

/// Reject a value where a §06 measure operation documents a measure.
///
/// Unenforced, `normalize(1)` typed a positive-integer SCALAR and
/// `weighted(true, 2)` the same, both at exit 0 — the measure operations thread
/// their argument's type through ([`fresh_measure`] on a non-measure yields the
/// argument back), so a scalar in gave a scalar out with no diagnostic and no
/// `%measure` anywhere. `truncate(false, interval(0, 1))` typed `%boolean` with a
/// unit-interval value set, which is worse: a boolean carrying a real support.
///
/// Refuses only a type that CANNOT be a measure — a scalar, an array, a
/// transposed vector, a record, a table, a tuple, an RNG state or a module
/// reference. A kernel and a `%deferred` operand both pass: §06's `normalize`
/// applies "on a non-nullary kernel", and a distribution with no type rule yet is
/// honestly `%deferred`, which this check must not turn into an error.
fn refuse_nonmeasure_operand(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
) -> Option<Type> {
    let (i, sentence) = measure_operand(name)?;
    let (node, ty, _) = args.get(i)?;
    let got = match ty {
        Type::Scalar(k) => format!("the {k} scalar"),
        Type::Array { shape, .. } => format!("a rank-{} array", shape.len()),
        Type::TVector { .. } => "a transposed vector".to_string(),
        Type::Record(fields) => format!("a {}-field record", fields.len()),
        Type::Table { columns, .. } => format!("a {}-column table", columns.len()),
        Type::Tuple(parts) => format!("a {}-component tuple", parts.len()),
        Type::RngState => "an RNG state".to_string(),
        Type::Module => "a module reference".to_string(),
        // Measure, Kernel, Function, Likelihood, and every open type: not decided
        // against, so not refused.
        _ => return None,
    };
    inf.diags.push(crate::Diagnostic::error_at(
        *node,
        format!(
            "`{name}` argument {} must be a measure or a kernel, got {got}: spec §06 \
             says {sentence}. Wrap a density in a measure first, as in \
             `weighted(f, Lebesgue(support = reals))`",
            i + 1,
        ),
    ));
    Some(Type::Failed(
        format!("{name}: argument {} is not a measure", i + 1).into(),
    ))
}

/// Explain the named reading only when this call actually spreads its argument.
const SPLAT_HINT: &str = " — this sole positional record or table splatted into one \
                          argument per field (spec §04), so its field names bind as \
                          argument names; to pass it as one ordinary argument use \
                          the keyword spelling, as in `f(pars = record(...))`";

/// Explain the positional reading when a tuple is spread over an ordered callable.
const TUPLE_SPLAT_HINT: &str = " — this sole positional tuple splatted into one positional \
                               argument per element (spec §04), because the callable does \
                               not accept a lone positional argument";

/// How a call's arguments read against a declared parameter count.
struct ArgReading {
    /// The argument count the reading supplies.
    count: usize,
    /// True when the call auto-splats a sole positional record, table or tuple,
    /// so that argument's FIELD names (record, table) or ELEMENTS (tuple) are what
    /// bind, not the keyword names.
    splatting: bool,
    /// True when the splatted argument is a tuple, which binds by position.
    tuple: bool,
}

impl ArgReading {
    /// Explain the spread reading only when this call actually spreads its argument.
    fn hint(&self) -> &'static str {
        if self.tuple {
            TUPLE_SPLAT_HINT
        } else if self.splatting {
            SPLAT_HINT
        } else {
            ""
        }
    }
}

/// Select the whole-object or spread reading from the input interface alone
/// (spec §04 "Calling conventions"). A callable that accepts a lone positional
/// argument (`takes_aggregate_whole`) receives a record, table or tuple whole.
/// Otherwise a record or table splats by name, and a tuple splats by position when
/// the callable has `ordered` inputs. An unresolved argument type defers a
/// possible spread, never changes the interface's decision after a domain check
/// fails.
fn arg_reading(
    inf: &Inferencer<'_, '_>,
    args: &[ArgInfo],
    named: &[NamedInfo],
    takes_aggregate_whole: bool,
    ordered: bool,
) -> Option<ArgReading> {
    let plain = args.len() + named.len();
    let plain_reading = Some(ArgReading {
        count: plain,
        splatting: false,
        tuple: false,
    });
    if !named.is_empty() || args.len() != 1 || takes_aggregate_whole {
        return plain_reading;
    }
    let (count, tuple) = match &args[0].1 {
        Type::Record(fields) => (fields.len(), false),
        Type::Table { columns, .. } => (columns.len(), false),
        Type::Tuple(parts) if ordered => (parts.len(), true),
        Type::Deferred | Type::Var(_) | Type::Any | Type::Failed(_)
            if matches!(inf.module.node(args[0].0), Node::Lit(_)) =>
        {
            return plain_reading;
        }
        Type::Deferred | Type::Var(_) | Type::Any | Type::Failed(_) => return None,
        // Not a record, table or spreadable tuple: an ordinary sole positional argument.
        _ => return plain_reading,
    };
    Some(ArgReading {
        count,
        splatting: true,
        tuple,
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
fn missing_required_args(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    declared: &[String],
    required: usize,
    reading: &ArgReading,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    // A tuple binds by position, so the arity check has already counted it.
    if (named.is_empty() && !reading.splatting) || reading.tuple {
        return None;
    }
    let supplied = supplied_arg_names(inf, reading, args, named);
    let supplied_positionals = if reading.splatting { 0 } else { args.len() };
    let missing = declared
        .iter()
        .take(required)
        .skip(supplied_positionals)
        .find(|name| !supplied.iter().any(|(supplied, _)| supplied == *name))?;
    inf.diags.push(crate::Diagnostic::error_at(
        id,
        format!("missing required argument `{missing}`"),
    ));
    Some(Type::Failed(
        format!("missing required argument `{missing}`").into(),
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
/// A call that spreads its aggregate also explains that name-binding step.
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
    let hint = reading.hint();
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
    if let Some(ty) = unresolved_shaped_argument(sig, args) {
        return Some(ty);
    }
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
        written_size: &|i| literal_size(&module.borrow(), args, i),
        intern: &|s| module.borrow_mut().intern(s),
    };
    let (ty, _) = lower(sig, &ctx);
    Some(ty)
}

/// The positive integer literal written at positional arg `i` as a static dim,
/// else `%dynamic`.
fn literal_size(module: &flatppl_core::Module, args: &[ArgInfo], i: usize) -> Dim {
    match args.get(i).map(|(node, _, _)| module.node(*node)) {
        Some(Node::Lit(Scalar::Int(n))) if *n > 0 => {
            u32::try_from(*n).map_or(Dim::Dynamic, Dim::Static)
        }
        _ => Dim::Dynamic,
    }
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
        written_size: &|i| literal_size(&module.borrow(), args, i),
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
    if matches!(name, "LKJ" | "LKJCholesky") {
        let n = named_or_positional_node(inf.module, named, args, "n", 0);
        return Some(lkj_matrix_domain(inf, n));
    }
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
        written_size: &|_| Dim::Dynamic,
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

    let kernel_node = resolve_binding_refs(inf, kernel_node);
    if let Node::Call(call) = inf.module.node(kernel_node)
        && matches!(call.head, CallHead::Builtin(op) if inf.module.resolve(op) == "broadcasted")
        && let [inner] = call.args.as_ref()
    {
        let inner = *inner;
        let cell = kernel_variate(inf, inner, None)?;
        if !matches!(cell, Type::Scalar(_)) {
            return None;
        }
        let input = resolve_binding_refs(inf, kernel_input_node?);
        let Type::Record(fields) = inf.lookup_type(input)? else {
            return None;
        };
        let mut shape: Option<Box<[Dim]>> = None;
        for (_, ty) in fields {
            match ty {
                Type::Scalar(_) => {}
                Type::Array { shape: s, elem } if matches!(elem.as_ref(), Type::Scalar(_)) => {
                    shape = Some(match shape {
                        None => s.clone(),
                        Some(previous) => broadcast_shape_join(&previous, s)?,
                    });
                }
                _ => return None,
            }
        }
        return Some(match shape {
            Some(shape) => Type::Array {
                shape,
                elem: Box::new(cell),
            },
            None => cell,
        });
    }

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
            written_size: &|_| Dim::Dynamic,
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
    if matches!(name.as_str(), "LKJ" | "LKJCholesky") {
        let input = kernel_input_node.map(|node| resolve_binding_refs(inf, node));
        let n = input.and_then(|node| match inf.module.node(node) {
            Node::Call(call)
                if matches!(call.head, CallHead::Builtin(op) if inf.module.resolve(op) == "record") =>
            {
                call.named
                    .iter()
                    .find(|field| inf.module.resolve(field.name) == "n")
                    .map(|field| field.value)
            }
            _ => None,
        });
        return Some(lkj_matrix_domain(inf, n));
    }
    lower_dist(inf, sig)
}

/// §08 LKJ and LKJCholesky produce n-by-n matrices. Retain the catalogue's
/// dynamic fallback when the existing fixed-size resolver cannot prove n.
fn lkj_matrix_domain(inf: &mut Inferencer<'_, '_>, n: Option<NodeId>) -> Type {
    let dimension = n.map_or(Dim::Dynamic, |node| resolve_dim(inf, node));
    Type::Array {
        shape: Box::new([dimension, dimension]),
        elem: Box::new(Type::Scalar(ScalarType::Real)),
    }
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

/// The whole support argument of a unary reference-measure constructor.
fn lebesgue_counting_support_node(
    inf: &Inferencer<'_, '_>,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<NodeId> {
    named
        .iter()
        .find(|(n, _, _, _)| inf.module.resolve(*n) == "support")
        .map(|(_, node, _, _)| *node)
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
    targets: Option<&LocalTargets>,
) -> RuleStep<ValueSet> {
    // User-callable application: the reified body's set rides over (for a
    // kernel call, the body set IS the output measure's support). A §09
    // standard-module reference has no reified body — its support/result set
    // is lowered from the catalogue sig with the call args.
    if let Some((callee_node, _)) = callee {
        if inf.module_catalogue_ref(*callee_node).is_some() {
            return Ready(catalogue_call_valueset(inf, *callee_node, args));
        }
        // Per-call substituted body value-set (arg sets bound to the callable's
        // parameters): a callable whose body tightens its range — `f(x) =
        // sqrt(x)` — carries `nonnegreals` to the call site. Fall back to the
        // un-substituted body set when substitution binds nothing or yields no
        // finer set.
        let callee_node = *callee_node;
        let step = substituted_result(inf, callee_node, args, named, targets, false);
        return step.map(inf, move |inf, sub| {
            if let Some((_, vs)) = sub
                && vs != ValueSet::Unknown
            {
                return vs;
            }
            match reified_body(inf, callee_node) {
                Some(body) => inf.lookup_valueset(body),
                None => ValueSet::Unknown,
            }
        });
    }
    let CallHead::Builtin(op) = call.head else {
        return Ready(ValueSet::Unknown);
    };
    // Reifications are callables, not values.
    if call.inputs.is_some() {
        return Ready(ValueSet::Unknown);
    }
    let name = inf.module.resolve(op).to_string();
    Ready(match name.as_str() {
        // A set-constructor used directly as a value binding is a PRESET (spec
        // §03): its value-set is the set it denotes. (Its TYPE is `%any` — a set
        // is not a value type — set in the `cartprod`/`interval`/… type arms.)
        "interval" | "stdsimplex" | "ordered" | "zerosum" | "cartpow" | "cartprod" => {
            set_call_valueset(inf, call)
        }
        // Parameters / loaded sets.
        "elementof" | "external" => set_expr_valueset(inf, args.first().map(|a| a.0)),
        "ksuperpose" => args
            .first()
            .map(|(head, ..)| kernel_support(inf, *head))
            .unwrap_or(ValueSet::Unknown),
        // `broadcast(head, data…)` over a user callable: the cell value-set is
        // the substituted body's set (per-cell arg sets bound to the head's
        // parameters), lifted into a `CartPow` over the result array. So
        // `broadcast(f, v)` with `f(x) = sqrt(x)` is `cartpow(nonnegreals, n)`.
        // Other heads (built-ins / §09 modules) fall back to the natural set.
        "broadcast" => {
            let Some((head_node, _, _)) = args.first() else {
                return Ready(ValueSet::Unknown);
            };
            let head_node = *head_node;
            let cell = |t: &Type| match t {
                Type::Array { elem, .. } => elem.as_ref().clone(),
                // A table's cells are its rows (§04 "Broadcasting").
                Type::Table { columns, .. } => Type::Record(columns.clone()),
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
            let dim = result_array_dim(ty);
            let step = substituted_result(inf, head_node, &cell_args, &cell_named, targets, false);
            return step.map(inf, move |_, sub| match (sub, dim) {
                (Some((_, cell_vs)), Some(dim)) if cell_vs != ValueSet::Unknown => {
                    ValueSet::CartPow(Box::new(cell_vs), dim)
                }
                _ => ValueSet::Unknown,
            });
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
        // `relabel` names the variate's entries without moving any (§06).
        "relabel" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        // The point mass at `value` lies in `value`'s set.
        "Dirac" => named_or_positional_node(inf.module, named, args, "value", 0)
            .map_or(ValueSet::Unknown, |n| inf.lookup_valueset(n)),
        // §06: the product measure's variate is the `cat` of the components', or a
        // record of them by name, so its support is the product of their supports.
        "joint" if matches!(ty, Type::Measure { .. }) => joint_support(inf, args, named),
        // §06: the variate is the `cat` (or record) of the first measure's and
        // each kernel's output variates, as with `joint`.
        "jointchain" if matches!(ty, Type::Measure { .. }) => {
            let mut parts: Vec<ValueSet> = Vec::with_capacity(args.len().max(named.len()));
            let mut names = Vec::with_capacity(named.len());
            let components: Vec<(Option<Symbol>, NodeId, Type)> = if named.is_empty() {
                args.iter().map(|(n, t, _)| (None, *n, t.clone())).collect()
            } else {
                named
                    .iter()
                    .map(|(s, n, t, _)| (Some(*s), *n, t.clone()))
                    .collect()
            };
            for (index, (name, node, component_ty)) in components.into_iter().enumerate() {
                let set = if index == 0 {
                    support_or_natural(inf, node, &component_ty)
                } else {
                    kernel_support(inf, node)
                };
                if set == ValueSet::Unknown {
                    return Ready(ValueSet::Unknown);
                }
                parts.push(set);
                names.extend(name);
            }
            product_support(parts, &names)
        }
        // §06: each trajectory entry is a draw from the kernel.
        "markovchain" | "kscan" => match (
            args.first()
                .map_or(ValueSet::Unknown, |(n, _, _)| kernel_support(inf, *n)),
            ty,
        ) {
            (ValueSet::Unknown, _) => ValueSet::Unknown,
            (cell, Type::Measure { domain, .. }) => match domain.as_ref() {
                Type::Array { shape, .. } if shape.len() == 1 => {
                    ValueSet::CartPow(Box::new(cell), shape[0])
                }
                Type::Table { nrows, .. } => ValueSet::CartPow(Box::new(cell), *nrows),
                _ => ValueSet::Unknown,
            },
            _ => ValueSet::Unknown,
        },
        // The image of a general function is not known statically.
        "pushfwd" => ValueSet::Unknown,
        // A sum of measures is supported on the union of their supports.
        "superpose" if matches!(ty, Type::Measure { .. }) => {
            let sets: Vec<ValueSet> = args
                .iter()
                .map(|(n, t, _)| support_or_natural(inf, *n, t))
                .collect();
            join_scalar_sets(&sets).unwrap_or(ValueSet::Unknown)
        }
        "truncate" => {
            // The truncated support lies inside both S and supp(M).
            let base = args
                .first()
                .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n));
            match set_expr_valueset(inf, args.get(1).map(|a| a.0)) {
                ValueSet::Unknown => base,
                set => intersect_intervals(&set, &base).unwrap_or(set),
            }
        }
        // A record literal lies in the product of its field sets, so
        // `lawof(record(a = a, b = b))` is the product of the draws' supports.
        // A field with no tighter set keeps its type's natural extent.
        "record" if args.is_empty() && !named.is_empty() => {
            let mut fields = Vec::with_capacity(named.len());
            for (name, node, field_ty, _) in named {
                let set = support_or_natural(inf, *node, field_ty);
                if set == ValueSet::Unknown {
                    return Ready(ValueSet::Unknown);
                }
                fields.push((*name, set));
            }
            ValueSet::RecordSet(fields.into())
        }
        // Reweighting never grows the support. `bayesupdate(L, prior)` keeps
        // the prior's.
        "normalize" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        "bayesupdate" => args
            .get(1)
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        "weighted" | "logweighted" | "tilt" => args
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
        "linsolve" | "qr" | "cross" => ValueSet::natural_of(ty),
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
        "conv" | "crosscorr" => ValueSet::natural_of(ty),
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
        // A §07 function row whose type is unresolved (`unresolved_shaped_argument`, or a
        // result threading an unresolved argument) has no set yet.
        _ if matches!(ty, Type::Deferred | Type::Failed(_))
            && matches!(
                crate::catalogue::builtin().base(&name),
                Some(crate::catalogue::Sig::Function { .. })
            ) =>
        {
            ValueSet::Unknown
        }
        _ => {
            if let Some(vs) = function_valueset(&mut *inf.module, &name, args) {
                vs
            } else {
                distribution_support(inf, &name, args, named)
            }
        }
    })
}

/// A node's value set, or for a measure its support, falling back to the
/// natural extent of its type.
fn support_or_natural(inf: &Inferencer<'_, '_>, node: NodeId, ty: &Type) -> ValueSet {
    match inf.lookup_valueset(node) {
        ValueSet::Unknown | ValueSet::Deferred => ValueSet::natural_of(ty),
        set => set,
    }
}

/// The support of `joint(...)` (§06): a record of the named components'
/// supports, the merged fields of record components, or the positional
/// product of scalar or vector components, which `cat` concatenates.
fn joint_support(inf: &Inferencer<'_, '_>, args: &[ArgInfo], named: &[NamedInfo]) -> ValueSet {
    if !named.is_empty() {
        let mut fields = Vec::with_capacity(named.len());
        for (name, node, ty, _) in named {
            match support_or_natural(inf, *node, ty) {
                ValueSet::Unknown => return ValueSet::Unknown,
                set => fields.push((*name, set)),
            }
        }
        return ValueSet::RecordSet(fields.into());
    }
    let sets: Vec<ValueSet> = args
        .iter()
        .map(|(n, t, _)| support_or_natural(inf, *n, t))
        .collect();
    product_support(sets, &[])
}

/// The product of component supports: a record when `names` names every
/// component, the merged fields of record components, or the positional
/// product, which `cat` concatenates (§06 `joint`).
fn product_support(sets: Vec<ValueSet>, names: &[Symbol]) -> ValueSet {
    if sets.is_empty() || sets.contains(&ValueSet::Unknown) {
        return ValueSet::Unknown;
    }
    if names.len() == sets.len() {
        return ValueSet::RecordSet(names.iter().copied().zip(sets).collect());
    }
    if sets.iter().all(|s| matches!(s, ValueSet::RecordSet(_))) {
        let fields: Vec<_> = sets
            .into_iter()
            .flat_map(|s| match s {
                ValueSet::RecordSet(fields) => fields.to_vec(),
                _ => Vec::new(),
            })
            .collect();
        return ValueSet::RecordSet(fields.into());
    }
    ValueSet::CartProd(sets.into())
}

/// A kernel's output support: its reified body's support, or the §08 support
/// of a bare distribution constructor used as a kernel.
fn kernel_support(inf: &mut Inferencer<'_, '_>, node: NodeId) -> ValueSet {
    if let Some(body) = reified_body(inf, node) {
        return inf.lookup_valueset(body);
    }
    match ksuperpose_constructor_name(inf, node) {
        Some(name) => distribution_support(inf, &name, &[], &[]),
        None => ValueSet::Unknown,
    }
}

/// `a ∩ b` when both are intervals or named real sets, elementwise through
/// matching `cartpow`, `cartprod` and records. `None` where the §03 vocabulary
/// has no form for the intersection; the caller then keeps the truncation set.
fn intersect_intervals(a: &ValueSet, b: &ValueSet) -> Option<ValueSet> {
    use ValueSet::*;
    fn limits(set: &ValueSet) -> Option<(f64, f64)> {
        match set {
            Reals => Some((f64::NEG_INFINITY, f64::INFINITY)),
            PosReals | NonNegReals => Some((0.0, f64::INFINITY)),
            UnitInterval => Some((0.0, 1.0)),
            Interval(lo, hi) => Some((*lo, *hi)),
            _ => None,
        }
    }
    if a.subset_of(b) {
        return Some(a.clone());
    }
    if b.subset_of(a) {
        return Some(b.clone());
    }
    match (a, b) {
        (CartPow(x, n), CartPow(y, m)) if n == m => {
            Some(CartPow(Box::new(intersect_intervals(x, y)?), *n))
        }
        (CartProd(xs), CartProd(ys)) if xs.len() == ys.len() => xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| intersect_intervals(x, y))
            .collect::<Option<Vec<_>>>()
            .map(|parts| CartProd(parts.into())),
        (RecordSet(xs), RecordSet(ys))
            if xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(x, y)| x.0 == y.0) =>
        {
            xs.iter()
                .zip(ys.iter())
                .map(|(x, y)| intersect_intervals(&x.1, &y.1).map(|s| (x.0, s)))
                .collect::<Option<Vec<_>>>()
                .map(|fields| RecordSet(fields.into()))
        }
        _ => {
            let ((a_lo, a_hi), (b_lo, b_hi)) = (limits(a)?, limits(b)?);
            let (lo, hi) = (a_lo.max(b_lo), a_hi.min(b_hi));
            (lo <= hi).then_some(Interval(lo, hi))
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
/// `stdsimplex(n)`, `ordered(n)`, `zerosum(n)`, `cartpow(S, size)`,
/// `cartprod(...)`). Shared by
/// `set_expr_valueset` (set-expression argument position) and `call_valueset`
/// (a set-constructor used directly as a preset binding, spec §03). `Unknown`
/// for any non-set-constructor head or unresolvable component.
fn set_call_valueset(inf: &mut Inferencer<'_, '_>, c: &Call) -> ValueSet {
    let mut budget = SET_EXPR_NODE_BUDGET;
    set_call_valueset_at(inf, c, &mut budget, 0)
}

fn set_call_valueset_at(
    inf: &mut Inferencer<'_, '_>,
    c: &Call,
    budget: &mut u32,
    depth: u32,
) -> ValueSet {
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
        "ordered" => ValueSet::Ordered(
            c.args
                .first()
                .map_or(Dim::Dynamic, |&n| resolve_dim(inf, n)),
        ),
        "zerosum" => ValueSet::ZeroSum(
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
            let elem = set_expr_valueset_at(inf, c.args.first().copied(), budget, depth + 1);
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
                    let set = set_expr_valueset_at(inf, Some(na.value), budget, depth + 1);
                    if set == ValueSet::Unknown {
                        return ValueSet::Unknown;
                    }
                    fields.push((na.name, set));
                }
                ValueSet::RecordSet(fields.into())
            } else {
                let mut parts = Vec::with_capacity(c.args.len());
                for &arg in c.args.iter() {
                    let set = set_expr_valueset_at(inf, Some(arg), budget, depth + 1);
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
pub(crate) fn set_expr_valueset(inf: &mut Inferencer<'_, '_>, node: Option<NodeId>) -> ValueSet {
    let mut budget = SET_EXPR_NODE_BUDGET;
    set_expr_valueset_at(inf, node, &mut budget, 0)
}

fn set_expr_valueset_at(
    inf: &mut Inferencer<'_, '_>,
    node: Option<NodeId>,
    budget: &mut u32,
    depth: u32,
) -> ValueSet {
    let Some(node) = node else {
        return ValueSet::Unknown;
    };
    // Work and stack are bounded separately; whichever runs out first defers.
    if *budget == 0 || depth > SET_EXPR_MAX_DEPTH {
        return ValueSet::Unknown;
    }
    *budget -= 1;
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
        Node::Call(c) => set_call_valueset_at(inf, &c, budget, depth),
        // A set bound to a name (§03 "Preset domains", §04 name reference) — the
        // reference denotes the same set as the binding's rhs.
        Node::Ref(r) if r.ns == RefNs::SelfMod => match inf.module.binding_by_name(r.name) {
            Some(b) => {
                let rhs = inf.module.binding(b).rhs;
                set_expr_valueset_at(inf, Some(rhs), budget, depth + 1)
            }
            None => ValueSet::Unknown,
        },
        Node::Ref(r)
            if matches!(r.ns, RefNs::Module(_)) && inf.module_set_element(node).is_some() =>
        {
            inf.lookup_valueset(node)
        }
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
    // A `BinCounts` support reads its edges by position, so bind keywords first.
    let declared = normalize_keyword_args(inf.module, name, args, named);
    let declared = declared.as_deref().unwrap_or(args);
    let ctx = LowerCtx {
        param_dim: &|kwarg| param_dim(inf, args, named, kwarg),
        arg_scalar: &|_| None,
        arg_dim: &|_| Dim::Dynamic,
        arg_type: &|i| arg_ty(declared, i).cloned(),
        written_size: &|_| Dim::Dynamic,
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
/// 2. The weights provably sum to one, by literals, a complement pair, or
///    exhaustive indexing of one simplex vector. No arithmetic
///    prover, so `superpose(weighted(w, m), weighted(1 - w, m2))` written with
///    two separately-bound halves of one sum is NOT proven.
/// 3. Every weight is provably in [0, 1], so each component is a measure and
///    the sum is a mixture rather than a signed combination.
fn superpose_is_provably_normalized(inf: &Inferencer<'_, '_>, args: &[ArgInfo]) -> bool {
    if args.is_empty() {
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
    literal_weights_sum_to_one(inf, &weights)
        || complement_pair(inf, &weights)
        || simplex_weights_cover_vector(inf, &weights)
}

/// §06: each coordinate of one simplex node occurs once, in any order.
fn simplex_weights_cover_vector(inf: &Inferencer<'_, '_>, weights: &[NodeId]) -> bool {
    let mut vector = None;
    let mut seen = vec![false; weights.len()];
    for &weight in weights {
        let Node::Call(call) = inf.module.node(resolve_binding_refs(inf, weight)) else {
            return false;
        };
        if !matches!(call.head, CallHead::Builtin(op) if inf.module.resolve(op) == "get")
            || call.args.len() != 2
            || !call.named.is_empty()
        {
            return false;
        }
        let base = resolve_binding_refs(inf, call.args[0]);
        if vector.is_some_and(|previous| previous != base)
            || !matches!(inf.lookup_valueset(base), ValueSet::StdSimplex(Dim::Static(n)) if n as usize == weights.len())
        {
            return false;
        }
        let Node::Lit(Scalar::Int(index)) = inf.module.node(call.args[1]) else {
            return false;
        };
        if *index < 1 || *index as usize > weights.len() || seen[*index as usize - 1] {
            return false;
        }
        seen[*index as usize - 1] = true;
        vector = Some(base);
    }
    !weights.is_empty()
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
pub(crate) fn resolve_binding_refs(inf: &Inferencer<'_, '_>, node: NodeId) -> NodeId {
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
/// §04 is explicit for the parametric case — `functionof` "traces the ancestor
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
        "draw"
            | "rand"
            | "elementof"
            | "external"
            | "load_data"
            | "rnginit"
            | "rngstate"
            | flatppl_core::INTEGRATION_POINT
    )
}

/// Does this expression subtree contain an [`is_opaque_value_source`] call?
///
/// Does NOT follow binding references, and that is the point: one binding is one
/// coordinate, so two `(%ref self psi)` nodes denote the same value however `psi`
/// was produced. Only sources written INSIDE the compared subtree defeat identity.
fn contains_opaque_value_source(inf: &Inferencer<'_, '_>, node: NodeId) -> bool {
    if let Node::Call(c) = inf.module.node(node)
        && let CallHead::Builtin(op) = c.head
        && is_opaque_value_source(inf.module.resolve(op))
    {
        return true;
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
/// parametric weight.
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
        // The broadcast rule already classified the product, even if deferred.
        "broadcast" | "ksuperpose" => mass,
        // §08 gives `intensity` as a "finite-mass measure or kernel". An infinite
        // intensity puts infinitely many events in some bin, so it is refused.
        "BinnedPoissonProcess" => {
            let intensity = named
                .iter()
                .find(|(n, ..)| inf.module.resolve(*n) == "intensity")
                .map(|(_, node, t, _)| (*node, t))
                .or_else(|| args.get(1).map(|(node, t, _)| (*node, t)));
            if let Some((
                node,
                Type::Measure {
                    mass: Mass::LocallyFinite,
                    ..
                },
            )) = intensity
            {
                inf.diags.push(crate::Diagnostic::error_at(
                    node,
                    "`BinnedPoissonProcess`'s `intensity` has infinite total mass, but \
                     spec §08 declares it a finite-mass measure or kernel",
                ));
                return Type::Failed("BinnedPoissonProcess: infinite intensity".into());
            }
            Mass::Normalized
        }
        // Every §08 distribution is a probability measure.
        // `Dirac(value)` is a point-mass probability measure (total mass 1).
        "Dirac" => Mass::Normalized,
        // Reference measures: finite on a bounded support, infinite (but
        // boundedly finite) on an unbounded one.
        "Lebesgue" | "Counting" => {
            let support_node = lebesgue_counting_support_node(inf, args, named);
            reference_mass(&set_expr_valueset(inf, support_node))
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

/// The mass of an independent product, pointwise for kernel components.
fn joint_mass(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo], named: &[NamedInfo]) -> Mass {
    let masses: Vec<Mass> = if !named.is_empty() {
        named
            .iter()
            .map(|(_, node, ty, _)| component_mass(inf, *node, ty))
            .collect()
    } else {
        args.iter()
            .map(|(node, ty, _)| component_mass(inf, *node, ty))
            .collect()
    };
    product_mass(&masses)
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
                    "kernelof" | "functionof" | "lawof" | "joint" | "ksuperpose" | "tilt"
                );
            }
            _ => return false,
        }
    }
}

/// Total-mass class of an independent product.
fn product_mass(masses: &[Mass]) -> Mass {
    use Mass::*;
    if masses.is_empty() {
        return Deferred;
    }
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
        Mass::Deferred => Mass::Deferred,
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
        "lengthof" => Some(Type::Scalar(Integer)),
        // scalar-real output
        // (divide and mean are NOT here: they are structural — divide promotes
        // its two operands, mean reduces to the array element type — handled
        // in call_rule, not the catalogue.)
        "logdensityof" | "densityof" => Some(Type::Scalar(Real)),
        "l1norm" | "l2norm" | "logsumexp" => Some(Type::Scalar(Real)),
        // §07 lists `posreals` ALONE for these four, with no `complexes` beside
        // it as `log` and `sqrt` carry, so the result is real for every argument.
        "log2" | "log10" | "gamma" | "loggamma" => Some(Type::Scalar(Real)),
        // scalar-complex output
        "cis" | "complex" => Some(Type::Scalar(Complex)),
        // scalar-boolean output
        "equal" | "unequal" | "lt" | "le" | "gt" | "ge" | "in" | "land" | "lor" | "lnot"
        | "isfinite" | "isinf" | "isnan" | "iszero" => Some(Type::Scalar(Boolean)),
        // real_or_complex: exp/log/sqrt/trig and friends
        "exp" | "log" | "sqrt" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh"
        | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" | "log1p" | "expm1" | "logit"
        | "invlogit" | "probit" | "invprobit" | "conj" => Some(real_or_cplx(arg0_scalar)),
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

#[cfg(test)]
mod no_laundering_tests {
    use super::*;

    /// The empty-product arm. `masses.iter().all(..)` is vacuously true on an
    /// empty slice, so a `product_mass` without this arm answers `Normalized`
    /// for a product of nothing — a definite class where no component supports
    /// one. `%deferred` is §11's "not yet inferred", which is the honest answer.
    #[test]
    fn an_empty_product_is_deferred_not_normalized() {
        assert_eq!(product_mass(&[]), Mass::Deferred);
    }

    /// A non-empty all-normalized product still answers `Normalized`, so the arm
    /// above is a carve-out for emptiness and not a blanket degrade.
    #[test]
    fn a_normalized_product_is_still_normalized() {
        assert_eq!(
            product_mass(&[Mass::Normalized, Mass::Normalized]),
            Mass::Normalized
        );
    }
}

#[cfg(test)]
mod placeholder_scope_tests {
    use super::*;

    /// §04 binds each `_` to the nearest enclosing `fn`. `fn(broadcast(fn(_ + 1), _))`
    /// desugars both holes to the same local name, so the outer body's placeholder refs
    /// must not include the inner `fn`'s, or substituting the outer argument types the
    /// inner `_ + 1` with it.
    #[test]
    fn a_nested_fn_keeps_its_own_placeholder() {
        let mut module = flatppl_syntax::parse("f = fn(broadcast(fn(_ + 1), _))").unwrap();
        let rhs = module.bindings().next().expect("one binding").1.rhs;
        let bundle = crate::ModuleBundle::new();
        let session = crate::modules::InferSession::new(&bundle);
        let inf = Inferencer::new(&mut module, Level::Type, &session);
        let (_, body) = local_reification(&inf, rhs).expect("`f` is a reification");
        let refs = collect_local_refs(&inf, body);
        let total: usize = refs.values().map(Vec::len).sum();
        assert_eq!(
            total, 1,
            "only the outer `_` is the outer body's placeholder: {refs:?}"
        );
    }
}
