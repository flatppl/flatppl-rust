//! Composite Gauss–Legendre integration with a fixed tensor workspace.
//! Returns the integral and its L1 magnitude for refinement certificates.

use super::*;

const GL_NODES: [f64; 16] = [
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878318,
    -0.755404408355003,
    -0.6178762444026438,
    -0.4580167776572274,
    -0.2816035507792589,
    -0.09501250983763744,
    0.09501250983763744,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.755404408355003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];
const GL_WEIGHTS: [f64; 16] = [
    0.02715245941175409,
    0.06225352393864789,
    0.09515851168249278,
    0.1246289712555339,
    0.1495959888165767,
    0.16915651939500254,
    0.1826034150449236,
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.16915651939500254,
    0.1495959888165767,
    0.1246289712555339,
    0.09515851168249278,
    0.06225352393864789,
    0.02715245941175409,
];

impl Emitter<'_> {
    pub(crate) fn gauss_legendre16(
        &mut self,
        upper: &Value,
        template: &Value,
        panels: &Value,
        workspace: u64,
        f: &mut impl FnMut(&mut Self, &Value) -> Value,
    ) -> (Value, Value) {
        let nodes: Vec<_> = GL_NODES.iter().map(|&v| self.scalar(v)).collect();
        let nodes = self.vector(&nodes);
        let weights: Vec<_> = GL_WEIGHTS.iter().map(|&v| self.scalar(v)).collect();
        let weights = self.vector(&weights);
        let sixteen = self.scalar(16.0);
        let half = self.scalar(0.5);
        let one = self.scalar(1.0);
        let count = self.mul(panels, &sixteen);
        let indices = self.index_vector(workspace, true);
        let indices = self.convert(&indices, ElemKind::Real);
        let active = self.compare("LT", &indices, &count);
        let last = self.sub(&count, &one);
        let safe_index = self.min(&indices, &last);
        let panel = self.div(&safe_index, &sixteen);
        let panel = self.floor(&panel);
        let offset = self.mul(&panel, &sixteen);
        let k = self.sub(&safe_index, &offset);
        let k = self.convert(&k, ElemKind::Int);
        let node = self.gather(&nodes, &k, 0);
        let weight = self.gather(&weights, &k, 0);
        let unit = self.add(&node, &one);
        let unit = self.mul(&unit, &half);
        let unit = self.add(&unit, &panel);
        let unit = self.div(&unit, panels);
        let batch = self.batch_rank(template);
        let mut dims = shape(&template.ty).to_vec();
        dims.push(Some(workspace));
        let axis = (dims.len() - 1) as u64;
        let axes = Axes {
            batch,
            layers: vec![1],
        };
        let ty = MlirTy::Ranked(dims);
        let unit = self.broadcast_in_dim(&unit, &[axis], ty.clone());
        let unit = self.axes_view(&unit, axes.clone());
        let weights = self.broadcast_in_dim(&weight, &[axis], ty.clone());
        let weights = self.axes_view(&weights, axes.clone());
        let active = self.broadcast_in_dim(&active, &[axis], ty);
        let active = self.axes_view(&active, axes);
        let t = self.mul(&unit, upper);
        let value = f(self, &t);
        let terms = self.mul(&weights, &value);
        let zero = self.constant_like(0.0, &terms);
        let terms = self.select(&active, &terms, &zero);
        let magnitude = self.abs(&terms);
        let magnitude = self.reduce_sum(&magnitude);
        let value = self.reduce_sum(&terms);
        let factor = self.mul(upper, &half);
        let factor = self.div(&factor, panels);
        let value = self.mul(&value, &factor);
        let magnitude = self.mul(&magnitude, &factor);
        let maximum = self.scalar(workspace as f64);
        let fits = self.compare("LE", &count, &maximum);
        let nan = self.scalar(f64::NAN);
        (
            self.select(&fits, &value, &nan),
            self.select(&fits, &magnitude, &nan),
        )
    }
}
