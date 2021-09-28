use ndarray::{Array, IxDyn, NdFloat, Zip};

use crate::variable::Module;
use crate::variable::VariableRef;

pub struct Relu {}

impl<T> Module<T> for Relu
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, _y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        x.mapv(|a| a.max(T::zero()))
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        left_ref: &'a VariableRef<T>,
        _right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        let mut grad = grad.clone();
        let ref data = left_ref.borrow().data;

        Zip::from(&mut grad).and(data).for_each(|g, &d| {
            *g = if d.is_sign_positive() { *g } else { T::zero() };
        });

        let zero = Array::<T, IxDyn>::zeros(grad.raw_dim());

        [grad, zero]
    }
}

impl<T> VariableRef<T>
where
    T: NdFloat,
{
    pub fn relu(&mut self) -> VariableRef<T> {
        let module = Relu {};
        module.subscribe(self, self, Box::new(Relu {}))
    }
}

#[cfg(test)]
mod tests {

    use crate::variable::Variable;
    use ndarray::array;

    #[test]
    fn check_method() {
        let mut x = Variable::new(array!([2.0]).into_dyn());

        assert_eq!(x.relu().borrow().data, array!([2.0]).into_dyn());

        let mut y = Variable::new(array!([-10.0]).into_dyn());

        assert_eq!(y.relu().borrow().data, array!([0.0]).into_dyn());
    }

    #[test]
    fn check_backward_positive() {
        let mut x = Variable::new(array!([2.0]).into_dyn());
        let mut z = x.relu();
        z.backward();
        assert_eq!(x.borrow().get_grad_f(), array!([1.0]).into_dyn());
    }

    #[test]
    fn check_backward_negative() {
        let mut x = Variable::new(array!([-2.0]).into_dyn());
        let mut z = x.relu();
        z.backward();
        assert_eq!(x.borrow().get_grad_f(), array!([0.0]).into_dyn());
    }
}
