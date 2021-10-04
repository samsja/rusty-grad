use ndarray::{Array, IxDyn, NdFloat};

use crate::variable::GradFn;
use crate::variable::VariableRef;

pub struct Identity {}

impl<T> GradFn<T> for Identity
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, _y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        x.clone()
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        _left_ref: &'a VariableRef<T>,
        _right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        let grad = grad.clone();
        let zero = Array::<T, IxDyn>::zeros(grad.raw_dim());

        [grad, zero]
    }
}

impl<T> VariableRef<T>
where
    T: NdFloat,
{
    pub fn identity(&mut self) -> VariableRef<T> {
        let grad_fn = Identity {};
        grad_fn.subscribe(self, self, Box::new(Identity {}))
    }
}
