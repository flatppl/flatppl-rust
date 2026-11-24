//! Real singular-value products on the smooth parts of the spectral maps.

use flatppl_core::graph::{build_call, build_record};
use flatppl_core::{Call, CallHead, Dim, Module, Node, NodeId, Scalar};

use crate::{Emitter, Step, real, record, tensor};

pub(super) fn operation(member: &str) -> Option<&'static str> {
    Some(match member {
        "svd" => "real svd",
        "cond" => "real condition number",
        "pinv" => "real pseudoinverse",
        "lstsq" => "real least squares",
        _ => return None,
    })
}

pub(super) fn structural(op: &str) -> bool {
    matches!(
        op,
        "real svd" | "real condition number" | "real pseudoinverse" | "real least squares"
    )
}

pub(super) fn valid_types(module: &Module, step: &Step<'_>) -> bool {
    let Some(&a) = step.args.first() else {
        return false;
    };
    if step.broadcast
        || !matches!(
            tensor::real_shape(module, a),
            Some([Dim::Static(1..), Dim::Static(1..)])
        )
    {
        return false;
    }
    let rhs_count = if step.op == "real least squares" {
        if !step
            .args
            .get(1)
            .is_some_and(|&b| matches!(tensor::real_shape(module, b), Some([_] | [_, _])))
        {
            return false;
        }
        2
    } else {
        1
    };
    step.args[rhs_count..]
        .iter()
        .all(|&arg| tensor::numeric_shape(module, arg) == Some(&[]))
}

pub(super) fn product(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
    forward: bool,
) -> Option<NodeId> {
    let base = step.args[0];
    let [Dim::Static(rows), Dim::Static(cols)] = *tensor::real_shape(&emitter.module, base)? else {
        return None;
    };
    let base = emitter.primal(base);
    let out = emitter.primal(step.node);
    let args: Vec<_> = step.args[1..]
        .iter()
        .map(|&arg| emitter.primal(arg))
        .collect();
    let Node::Call(call) = emitter.module.node(step.node) else {
        return None;
    };
    let CallHead::User(callee) = call.head else {
        return None;
    };
    let Node::Ref(callee) = emitter.module.node(callee) else {
        return None;
    };
    let namespace = callee.ns;
    let mut g = Graph {
        module: &mut emitter.module,
        namespace,
    };
    let k = rows.min(cols);
    let svd = if step.op == "real svd" {
        out
    } else {
        g.standard("svd", &[base])
    };
    let u = g.field(svd, "U");
    let s = g.field(svd, "S");
    let v = g.field(svd, "V");
    let f = Factors { u, s, v };
    if step.op == "real svd" {
        return Some(f.svd_product(&mut g, bar, forward, k));
    }
    if step.op == "real condition number" {
        return Some(f.condition_product(&mut g, bar, forward, k));
    }

    let least_squares = step.op == "real least squares";
    let (rhs, tolerance) = if least_squares {
        (Some(args[0]), &args[1..])
    } else {
        (None, args.as_slice())
    };
    let mut options = vec![base];
    options.extend_from_slice(tolerance);
    let cut = f.cutoff(&mut g, &options, k, rows.max(cols));
    let first_tolerance = if least_squares { 2 } else { 1 };
    if slot >= first_tolerance {
        let valid = if slot == first_tolerance {
            let ordinary = g.op("lnot", &[cut.automatic]);
            g.op("land", &[cut.valid, ordinary])
        } else {
            cut.valid
        };
        let zero = g.real(0.0);
        let result = if forward {
            g.map("mul", &[out, zero])
        } else {
            zero
        };
        return Some(g.guard(valid, result, forward));
    }
    if slot == 1 {
        let rhs_map = g.standard("pinv", &options);
        let map = if forward { rhs_map } else { g.t(rhs_map) };
        return Some(g.mul(map, bar));
    }
    let direction = if let Some(rhs) = rhs.filter(|_| !forward) {
        // A vector RHS uses an outer product, a matrix RHS contracts columns.
        g.mul_t(bar, rhs)
    } else {
        bar
    };
    let result = f.inverse_product(&mut g, direction, cut, forward, k);
    Some(if let Some(rhs) = rhs.filter(|_| forward) {
        g.mul(result, rhs)
    } else {
        result
    })
}

pub(super) fn mask(
    emitter: &mut Emitter,
    step: &Step<'_>,
    slot: usize,
    bar: NodeId,
    derivative: NodeId,
) -> Option<NodeId> {
    let array = !tensor::real_shape(&emitter.module, step.args[slot])?.is_empty();
    let m = &mut emitter.module;
    let inactive = if step.op == "real condition number" {
        build_call(m, "iszero", &[bar])
    } else if step.op == "real svd" {
        let mut zero = m.alloc(Node::Lit(Scalar::Bool(true)));
        for name in ["U", "S", "V"] {
            let symbol = m.intern(name);
            let component = record::project(m, bar, symbol);
            let component = tensor::broadcast(m, "iszero", &[component]);
            let component = build_call(m, "lall", &[component]);
            zero = build_call(m, "land", &[zero, component]);
        }
        zero
    } else {
        let zero = tensor::broadcast(m, "iszero", &[bar]);
        build_call(m, "lall", &[zero])
    };
    let zero = real(m, 0.0);
    let call = if array { tensor::broadcast } else { build_call };
    Some(call(m, "ifelse", &[inactive, zero, derivative]))
}

struct Graph<'a> {
    module: &'a mut Module,
    namespace: flatppl_core::RefNs,
}

impl Graph<'_> {
    fn op(&mut self, name: &str, args: &[NodeId]) -> NodeId {
        build_call(self.module, name, args)
    }

    fn map(&mut self, name: &str, args: &[NodeId]) -> NodeId {
        tensor::broadcast(self.module, name, args)
    }

    fn real(&mut self, value: f64) -> NodeId {
        real(self.module, value)
    }

    fn integer(&mut self, value: u32) -> NodeId {
        tensor::int(self.module, i64::from(value))
    }

    fn index(&mut self, a: NodeId, i: u32) -> NodeId {
        let i = self.integer(i);
        self.op("get", &[a, i])
    }

    fn field(&mut self, rec: NodeId, name: &str) -> NodeId {
        let name = self.module.intern(name);
        record::project(self.module, rec, name)
    }

    fn record(&mut self, fields: &[(&str, NodeId)]) -> NodeId {
        let fields: Vec<_> = fields
            .iter()
            .map(|&(name, value)| (self.module.intern(name), value))
            .collect();
        build_record(self.module, &fields)
    }

    fn standard(&mut self, name: &str, args: &[NodeId]) -> NodeId {
        let name = self.module.intern(name);
        let callee = self.module.alloc(Node::Ref(flatppl_core::Ref {
            ns: self.namespace,
            name,
        }));
        self.module.alloc(Node::Call(Call {
            head: CallHead::User(callee),
            args: args.into(),
            named: Box::new([]),
            inputs: None,
        }))
    }

    fn axis(&mut self, a: NodeId, row: bool) -> NodeId {
        let zero = self.integer(0);
        let one = self.integer(1);
        let (before, after) = if row { (one, zero) } else { (zero, one) };
        self.op("addaxes", &[a, before, after])
    }

    fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.op("mul", &[a, b])
    }

    fn t(&mut self, a: NodeId) -> NodeId {
        self.op("transpose", &[a])
    }

    fn mul_t(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let b = self.t(b);
        self.mul(a, b)
    }

    fn t_mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a = self.t(a);
        self.mul(a, b)
    }

    fn columns(&mut self, a: NodeId, scale: NodeId) -> NodeId {
        let row = self.axis(scale, true);
        self.map("mul", &[a, row])
    }

    fn guard(&mut self, ok: NodeId, result: NodeId, array: bool) -> NodeId {
        let nan = self.real(f64::NAN);
        if array {
            self.map("ifelse", &[ok, result, nan])
        } else {
            self.op("ifelse", &[ok, result, nan])
        }
    }

    fn complement(&mut self, basis: NodeId, a: NodeId) -> NodeId {
        let projection = self.t_mul(basis, a);
        let projection = self.mul(basis, projection);
        self.op("sub", &[a, projection])
    }

    fn off_diagonal(&mut self, k: u32) -> NodeId {
        let labels: Vec<_> = (1..=k).map(|i| self.integer(i)).collect();
        let labels = self.op("vector", &labels);
        let l = self.axis(labels, false);
        let r = self.axis(labels, true);
        self.map("unequal", &[l, r])
    }
}

struct Factors {
    u: NodeId,
    s: NodeId,
    v: NodeId,
}

struct Cutoff {
    reciprocal: NodeId,
    retained: NodeId,
    valid: NodeId,
    automatic: NodeId,
}

impl Factors {
    fn core(&self, g: &mut Graph<'_>, e: NodeId, inverse: bool) -> NodeId {
        let (left, right) = if inverse {
            (self.v, self.u)
        } else {
            (self.u, self.v)
        };
        let projected = g.t_mul(left, e);
        g.mul(projected, right)
    }

    fn expand(&self, g: &mut Graph<'_>, e: NodeId, inverse: bool) -> NodeId {
        let (left, right) = if inverse {
            (self.v, self.u)
        } else {
            (self.u, self.v)
        };
        let projected = g.mul(left, e);
        g.mul_t(projected, right)
    }

    fn svd_product(&self, g: &mut Graph<'_>, bar: NodeId, forward: bool, k: u32) -> NodeId {
        let zero = g.real(0.0);
        let one = g.real(1.0);
        let si = g.axis(self.s, false);
        let sj = g.axis(self.s, true);
        // Use (s_j - s_i)(s_j + s_i) rather than subtracting two squares.
        let diff = g.map("sub", &[sj, si]);
        let total = g.map("add", &[sj, si]);
        let gap = g.map("mul", &[diff, total]);
        let independent = g.map("iszero", &[diff]);
        let distinct = g.map("lnot", &[independent]);
        let off = g.off_diagonal(k);
        let diagonal = g.map("lnot", &[off]);
        let regular = g.map("lor", &[diagonal, distinct]);
        let regular = g.op("lall", &[regular]);
        let smallest = g.index(self.s, k);
        let positive = g.op("gt", &[smallest, zero]);
        let valid = g.op("land", &[regular, positive]);
        let has_gap = g.map("land", &[distinct, off]);
        let safe_gap = g.map("ifelse", &[has_gap, gap, one]);
        let reciprocal = g.map("divide", &[one, self.s]);
        if forward {
            let c = self.core(g, bar, false);
            let ct = g.t(c);
            let a = g.map("mul", &[sj, c]);
            let b = g.map("mul", &[si, ct]);
            let left = g.op("add", &[a, b]);
            let a = g.map("mul", &[si, c]);
            let b = g.map("mul", &[sj, ct]);
            let right = g.op("add", &[a, b]);
            let mut vectors = Vec::new();
            let transposed = g.t(bar);
            for (num, basis, opposite, e) in [
                (left, self.u, self.v, bar),
                (right, self.v, self.u, transposed),
            ] {
                let skew = g.map("divide", &[num, safe_gap]);
                let skew = g.map("ifelse", &[has_gap, skew, zero]);
                let parallel = g.mul(basis, skew);
                let component = g.mul(e, opposite);
                let component = g.complement(basis, component);
                let component = g.columns(component, reciprocal);
                let total = g.op("add", &[parallel, component]);
                vectors.push(g.guard(valid, total, true));
            }
            let ds = g.op("diag", &[c]);
            let ds = g.guard(valid, ds, true);
            g.record(&[("U", vectors[0]), ("S", ds), ("V", vectors[1])])
        } else {
            let ub = g.field(bar, "U");
            let sb = g.field(bar, "S");
            let vb = g.field(bar, "V");
            let mu = g.t_mul(self.u, ub);
            let mv = g.t_mul(self.v, vb);
            let mut parts = Vec::new();
            for (matrix, scale) in [(mu, sj), (mv, si)] {
                let mt = g.t(matrix);
                let skew = g.op("sub", &[matrix, mt]);
                let scaled = g.map("mul", &[scale, skew]);
                let scaled = g.map("divide", &[scaled, safe_gap]);
                parts.push(g.map("ifelse", &[has_gap, scaled, zero]));
            }
            let main = g.op("add", &parts);
            let diag = g.op("diagmat", &[sb]);
            let main = g.op("add", &[main, diag]);
            let main = self.expand(g, main, false);
            let left = g.complement(self.u, ub);
            let left = g.columns(left, reciprocal);
            let left = g.mul_t(left, self.v);
            let right = g.complement(self.v, vb);
            let right = g.columns(right, reciprocal);
            let right = g.mul_t(self.u, right);
            let result = g.op("add", &[main, left]);
            let result = g.op("add", &[result, right]);
            g.guard(valid, result, true)
        }
    }

    fn condition_product(&self, g: &mut Graph<'_>, bar: NodeId, forward: bool, k: u32) -> NodeId {
        let max = g.index(self.s, 1);
        let min = g.index(self.s, k);
        let zero = g.real(0.0);
        let positive = g.op("gt", &[min, zero]);
        let valid = if k > 1 {
            let second = g.index(self.s, 2);
            let before = g.index(self.s, k - 1);
            let top = g.op("gt", &[max, second]);
            let bottom = g.op("gt", &[before, min]);
            let simple = g.op("land", &[top, bottom]);
            g.op("land", &[positive, simple])
        } else {
            positive
        };
        let all = g.module.intern("all");
        let colon = g.module.alloc(Node::Const(all));
        let first = g.integer(1);
        let last = g.integer(k);
        let i = g.op("get", &[self.u, colon, first]);
        let j = g.op("get", &[self.v, colon, first]);
        let l = g.op("get", &[self.u, colon, last]);
        let r = g.op("get", &[self.v, colon, last]);
        let top = g.mul_t(i, j);
        let bottom = g.mul_t(l, r);
        let ratio = g.op("divide", &[max, min]);
        let bottom = g.mul(bottom, ratio);
        let difference = g.op("sub", &[top, bottom]);
        let local = g.map("divide", &[difference, min]);
        let weighted = g.map("mul", &[local, bar]);
        let result = if forward {
            g.op("sum", &[weighted])
        } else {
            weighted
        };
        g.guard(valid, result, !forward)
    }

    fn cutoff(&self, g: &mut Graph<'_>, args: &[NodeId], k: u32, size: u32) -> Cutoff {
        let rank = g.standard("rank", args);
        let labels: Vec<_> = (1..=k).map(|i| g.integer(i)).collect();
        let labels = g.op("vector", &labels);
        let retained = g.map("le", &[labels, rank]);
        let real_rank = g.op("real", &[rank]);
        let one = g.real(1.0);
        let zero = g.real(0.0);
        let safe = g.map("ifelse", &[retained, self.s, one]);
        let inverse = g.map("divide", &[one, safe]);
        let reciprocal = g.map("ifelse", &[retained, inverse, zero]);
        let largest = g.index(self.s, 1);
        let sentinel = g.real(-1.0);
        let relative = args.get(1).copied().unwrap_or(sentinel);
        let absolute = args.get(2).copied().unwrap_or(zero);
        let shifted = g.op("sub", &[relative, sentinel]);
        let automatic = g.op("iszero", &[shifted]);
        let threshold = g.mul(relative, largest);
        let threshold = g.op("max", &[threshold, absolute]);
        let regular = self.away_from(g, threshold);
        // The backend rank applies the actual precision-dependent cutoff. A
        // candidate threshold can be a switching point only if it selects that rank.
        let mut default = g.module.alloc(Node::Lit(Scalar::Bool(true)));
        for eps in [f32::EPSILON as f64, f64::EPSILON] {
            let rel = g.real(eps * f64::from(size));
            let threshold = g.mul(rel, largest);
            let threshold = g.op("max", &[threshold, absolute]);
            let retained = g.map("gt", &[self.s, threshold]);
            let indicators = g.map("real", &[retained]);
            let count = g.op("sum", &[indicators]);
            let difference = g.op("sub", &[count, real_rank]);
            let same = g.op("iszero", &[difference]);
            let different = g.op("lnot", &[same]);
            let away = self.away_from(g, threshold);
            let regular = g.op("lor", &[different, away]);
            default = g.op("land", &[default, regular]);
        }
        let valid = g.op("ifelse", &[automatic, default, regular]);
        Cutoff {
            reciprocal,
            retained,
            valid,
            automatic,
        }
    }

    fn away_from(&self, g: &mut Graph<'_>, boundary: NodeId) -> NodeId {
        let difference = g.map("sub", &[self.s, boundary]);
        let tied = g.map("iszero", &[difference]);
        let away = g.map("lnot", &[tied]);
        g.op("lall", &[away])
    }

    fn inverse_product(
        &self,
        g: &mut Graph<'_>,
        bar: NodeId,
        cut: Cutoff,
        forward: bool,
        k: u32,
    ) -> NodeId {
        let c = self.core(g, bar, !forward);
        let zero = g.real(0.0);
        let one = g.real(1.0);
        let di = g.axis(cut.reciprocal, false);
        let dj = g.axis(cut.reciprocal, true);
        let si = g.axis(self.s, false);
        let sj = g.axis(self.s, true);
        let ri = g.axis(cut.retained, false);
        let rj = g.axis(cut.retained, true);
        let both = g.map("land", &[ri, rj]);
        let mixed = g.map("lxor", &[ri, rj]);
        let diff = g.map("sub", &[sj, si]);
        let total = g.map("add", &[sj, si]);
        let gap = g.map("mul", &[diff, total]);
        let safe = g.map("ifelse", &[mixed, gap, one]);

        // In singular coordinates the derivative is a pair of entrywise maps.
        // Equal singular values on the same side of the cutoff use their limits.
        let a = g.map("mul", &[dj, si]);
        let b = g.map("mul", &[di, sj]);
        let difference = g.map("sub", &[a, b]);
        let cross = g.map("divide", &[difference, safe]);
        let cross = g.map("ifelse", &[mixed, cross, zero]);
        let kept = g.map("mul", &[di, dj]);
        let kept = g.op("neg", &[kept]);
        let direct = g.map("ifelse", &[both, kept, cross]);
        let a = g.map("mul", &[dj, sj]);
        let b = g.map("mul", &[di, si]);
        let difference = g.map("sub", &[a, b]);
        let cross = g.map("divide", &[difference, safe]);
        let cross = g.map("ifelse", &[mixed, cross, zero]);
        let off = g.off_diagonal(k);
        let cross = g.map("ifelse", &[off, cross, zero]);
        let term = if forward {
            let c = g.t(c);
            g.map("mul", &[cross, c])
        } else {
            let weighted = g.map("mul", &[cross, c]);
            g.t(weighted)
        };
        let main = g.map("mul", &[direct, c]);
        let main = g.op("add", &[main, term]);
        let main = self.expand(g, main, forward);

        let scale = g.map("mul", &[cut.reciprocal, cut.reciprocal]);
        let transpose = g.t(bar);
        let (left, right) = if forward {
            (self.v, self.u)
        } else {
            (self.u, self.v)
        };
        let a = g.mul(transpose, right);
        let a = g.complement(left, a);
        let a = g.columns(a, scale);
        let a = g.mul_t(a, right);
        let b = g.mul(bar, left);
        let b = g.complement(right, b);
        let b = g.columns(b, scale);
        let b = g.mul_t(left, b);
        let result = g.op("add", &[main, a]);
        let result = g.op("add", &[result, b]);
        g.guard(cut.valid, result, true)
    }
}
