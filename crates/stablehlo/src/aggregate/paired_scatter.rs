//! Fuse a word-masked quotient followed by an additive row scatter.
use flatppl_core::{Dim, Inputs, Node, NodeId, Scalar, ScalarType, Type};

use crate::emitter::Emitter;
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::refuse::EmitError;

use super::{RowScatter, builtin_args};

pub(super) struct PairedScatter {
    matrix: NodeId,
    gather: NodeId,
    docs: NodeId,
    words: NodeId,
    seed: NodeId,
    rows: u64,
    columns: u64,
    doc_base: i64,
    word_base: i64,
}

impl PairedScatter {
    pub(super) fn read(e: &Emitter, row: &RowScatter) -> Option<Self> {
        let [updates] = *args(e, row.updates, "rowstack")? else {
            return None;
        };
        let [mask, quotient, zero] = *dotted(e, full_rows_matrix(e, updates)?, "ifelse")? else {
            return None;
        };
        let [lhs, rhs] = *dotted(e, mask, "equal")? else {
            return None;
        };
        let [zero_seed, seed_zero, division] = *dotted(e, quotient, "ifelse")? else {
            return None;
        };
        if !is_positive_zero(e, zero) || !is_positive_zero(e, seed_zero) {
            return None;
        }
        let [numerator, denominator] = *dotted(e, division, "divide")? else {
            return None;
        };
        let [tested] = *dotted(e, zero_seed, "iszero")? else {
            return None;
        };
        if resolve(e, numerator) != resolve(e, tested) {
            return None;
        }
        let seed = added_axes(e, numerator, 0, 1)?;
        let (words, word_base) =
            word_mask(e, lhs, rhs, denominator).or_else(|| word_mask(e, rhs, lhs, denominator))?;
        let [gather] = *args(e, denominator, "rowstack")? else {
            return None;
        };
        let doc_op = if row.base == 0 { "get0" } else { "get" };
        let [nested, docs] = *args(e, gather, doc_op)? else {
            return None;
        };
        if resolve(e, docs) != resolve(e, row.indices) {
            return None;
        }
        let matrix = full_rows_matrix(e, nested)?;
        let [Dim::Static(rows), Dim::Static(columns)] = *array_shape(e, matrix, ScalarType::Real)?
        else {
            return None;
        };
        let [Dim::Static(count)] = *array_shape(e, docs, ScalarType::Integer)? else {
            return None;
        };
        if row.len != u64::from(rows)
            || rows == 0
            || columns == 0
            || count == 0
            || u64::from(rows) * u64::from(columns) > i32::MAX as u64
            || array_shape(e, words, ScalarType::Integer)? != [Dim::Static(count)]
            || array_shape(e, seed, ScalarType::Real)? != [Dim::Static(count)]
        {
            return None;
        }
        Some(Self {
            matrix,
            gather,
            docs,
            words,
            seed,
            rows: rows.into(),
            columns: columns.into(),
            doc_base: row.base,
            word_base,
        })
    }

    pub(super) fn lower(self, e: &mut Emitter) -> Result<Option<Value>, EmitError> {
        let raw_docs = e.lower_node(self.docs)?;
        let raw_words = e.lower_node(self.words)?;
        let seed = e.lower_node(self.seed)?;
        let matrix = e.lower_node(self.matrix)?;
        if raw_docs.elem != ElemKind::Int
            || raw_words.elem != ElemKind::Int
            || seed.elem != ElemKind::Real
            || matrix.elem != ElemKind::Real
            || [&raw_docs, &raw_words, &seed, &matrix]
                .iter()
                .any(|value| e.batch_rank(value) != 0)
            || !matches!(&raw_docs.ty, MlirTy::Ranked(dims) if matches!(dims.as_slice(), [Some(_)]))
            || raw_docs.ty != raw_words.ty
            || raw_docs.ty != seed.ty
            || matrix.ty != MlirTy::Ranked(vec![Some(self.rows), Some(self.columns)])
        {
            return Ok(None);
        }
        // Preserve the denominator gather's checked row semantics. Missing
        // words only control the equality mask and contribute no update.
        let checked_docs = e.checked_index(self.gather, &raw_docs, self.rows, self.doc_base)?;
        let valid_docs = e.compare("EQ", &raw_docs, &checked_docs);
        let first_word = e.int_value_const(self.word_base);
        let last_word = e.int_value_const(self.word_base + self.columns as i64 - 1);
        let after_first = e.compare("GE", &raw_words, &first_word);
        let before_last = e.compare("LE", &raw_words, &last_word);
        let valid_words = e.and(&after_first, &before_last);
        let checked_words = e.select(&valid_words, &raw_words, &first_word);
        let first_doc = e.int_value_const(self.doc_base);
        let docs = e.sub(&checked_docs, &first_doc);
        let words = e.sub(&checked_words, &first_word);
        let columns = e.int_value_const(self.columns as i64);
        let offsets = e.mul(&docs, &columns);
        let linear = e.add(&offsets, &words);

        let cells = self.rows * self.columns;
        let flat = e.reshape(&matrix, MlirTy::Ranked(vec![Some(cells)]));
        let selected = e.gather(&flat, &linear, 0);
        let quotient = e.div(&seed, &selected);
        let zero = e.scalar(0.0);
        let zero_seed = e.compare("EQ", &seed, &zero);
        let updates = e.select(&zero_seed, &zero, &quotient);
        let valid = e.and(&valid_docs, &valid_words);
        let missing = e.int_value_const(-1);
        let scatter_indices = e.select(&valid, &linear, &missing);
        let scattered = e.scatter_add_rows(cells, &scatter_indices, &updates);
        Ok(Some(e.reshape(&scattered, matrix.ty)))
    }
}

// A row lambda over the full ordered axis is a view of its held matrix.
fn full_rows_matrix(e: &Emitter, node: NodeId) -> Option<NodeId> {
    let [callback, range, held] = *args(e, node, "broadcast")? else {
        return None;
    };
    let [matrix] = *args(e, held, "vector")? else {
        return None;
    };
    let [Dim::Static(rows), _] = *array_shape(e, matrix, ScalarType::Real)? else {
        return None;
    };
    if !matches!(args(e, range, "indicesof"), Some([_]))
        || array_shape(e, range, ScalarType::Integer)? != [Dim::Static(rows)]
    {
        return None;
    }
    let callback = resolve(e, callback);
    let Node::Call(call) = e.node(callback) else {
        return None;
    };
    let [body] = *builtin_args(e, callback, "functionof")? else {
        return None;
    };
    let Some(Inputs::Spec(parameters)) = &call.inputs else {
        return None;
    };
    let [(_, row_parameter), (_, matrix_parameter)] = parameters.as_ref() else {
        return None;
    };
    let [m, r, all] = *builtin_args(e, body, "get")? else {
        return None;
    };
    if !matches!(e.node(m), Node::Ref(r) if r == matrix_parameter)
        || !matches!(e.node(r), Node::Ref(r) if r == row_parameter)
        || !matches!(e.node(all), Node::Const(s) if e.resolve(*s) == "all")
    {
        return None;
    }
    Some(matrix)
}

fn word_mask(
    e: &Emitter,
    words: NodeId,
    range: NodeId,
    denominator: NodeId,
) -> Option<(NodeId, i64)> {
    let words = added_axes(e, words, 0, 1)?;
    let range = added_axes(e, range, 1, 0)?;
    let (row, base) = if let Some(&[row]) = args(e, range, "indicesof") {
        (row, 1)
    } else {
        let [row] = *args(e, range, "indicesof0")? else {
            return None;
        };
        (row, 0)
    };
    let [matrix, first] = *args(e, row, "get")? else {
        return None;
    };
    (resolve(e, matrix) == resolve(e, denominator)
        && matches!(e.node(first), Node::Lit(Scalar::Int(1))))
    .then_some((words, base))
}

fn added_axes(e: &Emitter, node: NodeId, leading: i64, trailing: i64) -> Option<NodeId> {
    let [input, left, right] = *args(e, node, "addaxes")? else {
        return None;
    };
    (matches!(e.node(left), Node::Lit(Scalar::Int(n)) if *n == leading)
        && matches!(e.node(right), Node::Lit(Scalar::Int(n)) if *n == trailing))
    .then_some(input)
}

fn is_positive_zero(e: &Emitter, node: NodeId) -> bool {
    matches!(e.node(resolve(e, node)), Node::Lit(Scalar::Real(x)) if x.to_bits() == 0)
}

fn array_shape<'m>(e: &'m Emitter, node: NodeId, kind: ScalarType) -> Option<&'m [Dim]> {
    let Type::Array { shape, elem } = e.type_of(node)? else {
        return None;
    };
    (**elem == Type::Scalar(kind)).then_some(shape)
}

fn resolve(e: &Emitter, node: NodeId) -> NodeId {
    e.resolve_tuple_projections(node)
}

fn args<'m>(e: &'m Emitter, node: NodeId, head: &str) -> Option<&'m [NodeId]> {
    builtin_args(e, resolve(e, node), head)
}

fn dotted<'m>(e: &'m Emitter, node: NodeId, head: &str) -> Option<&'m [NodeId]> {
    let (operator, operands) = args(e, node, "broadcast")?.split_first()?;
    matches!(e.node(*operator), Node::Const(symbol) if e.resolve(*symbol) == head)
        .then_some(operands)
}
