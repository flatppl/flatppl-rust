//! A checked runtime partition shared by portable family summaries.

use super::grouped::{fixed_indices, gather_parts};
use super::regression::setup;
use super::{Expr, Factor, Reducer, map, materialize, num, op, resolve};
use flatppl_core::{Dim, Module, NodeId, ScalarType, Symbol, Type};

pub(super) fn size(m: &Module, id: NodeId, kind: ScalarType) -> Option<u32> {
    match m.type_of(id)? {
        Type::Array { shape, elem } if **elem == Type::Scalar(kind) => match **shape {
            [Dim::Static(n)] => Some(n),
            _ => None,
        },
        _ => None,
    }
}

pub(super) struct Partition {
    pub parameters: Vec<(Symbol, NodeId, bool)>,
    pub indices: NodeId,
    pub origin: i64,
    pub groups: u32,
    pub count: u32,
}

impl Partition {
    pub fn new(m: &Module, f: &Factor, r: &Reducer, kind: ScalarType) -> Option<Self> {
        let count = size(m, f.observed, kind)?;
        if count == 0 || !r.data_only(m, f.observed) {
            return None;
        }
        let mut partition = None;
        let mut parameters = Vec::new();
        for &(name, value) in &f.parameters {
            if let Some(value) = r.scalar(m, value) {
                parameters.push((name, value, false));
                continue;
            }
            let (array, indices, origin) = gather_parts(m, value, r)?;
            let groups = size(m, array, ScalarType::Real)?;
            let indices = resolve(m, indices, &r.boundary);
            if partition.is_some_and(|(previous, base, extent)| {
                base != origin
                    || extent != groups
                    || r.data_key(m, previous) != r.data_key(m, indices)
            }) || size(m, indices, ScalarType::Integer)? != count
                || !r.data_only(m, indices)
                || fixed_indices(m, indices, r).is_some()
            {
                return None;
            }
            partition = Some((indices, origin, groups));
            parameters.push((name, array, true));
        }
        let (indices, origin, groups) = partition?;
        if !(1..=64).contains(&groups) || u64::from(count) * u64::from(groups) > 1 << 22 {
            return None;
        }
        Some(Self {
            parameters,
            indices,
            origin,
            groups,
            count,
        })
    }

    /// Check the original gather first, then normalize to one-based labels.
    pub fn checked(&self, m: &mut Module) -> Expr {
        let labels = op(
            if self.origin == 0 {
                "indicesof0"
            } else {
                "indicesof"
            },
            [op(
                "fill",
                [Expr::Integer(0), Expr::Integer(i64::from(self.groups))],
            )],
        );
        setup(
            m,
            map(
                "add",
                [
                    op(
                        if self.origin == 0 { "get0" } else { "get" },
                        [labels, Expr::Node(self.indices)],
                    ),
                    Expr::Integer(1 - self.origin),
                ],
            ),
        )
    }

    pub fn keys(&self, m: &mut Module, observed: NodeId) -> [NodeId; 4] {
        [
            observed,
            self.indices,
            materialize(m, Expr::Integer(i64::from(self.groups))),
            materialize(m, Expr::Integer(self.origin)),
        ]
    }

    pub fn counts(&self, r: &mut Reducer, m: &mut Module, checked: Expr) -> Expr {
        let keys = self.keys(m, self.indices);
        r.summary_for(
            m,
            &keys[1..],
            "runtime_group_counts",
            op(
                "bincounts",
                [
                    op("vector", (0..=self.groups).map(|i| num(f64::from(i) + 0.5))),
                    checked,
                ],
            ),
        )
    }

    pub fn axes(&self) -> Expr {
        op(
            "indicesof",
            [op(
                "fill",
                [Expr::Integer(0), Expr::Integer(i64::from(self.groups))],
            )],
        )
    }
}
