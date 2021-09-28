use ndarray::{Array, Dimension, NdFloat, Zip};

use crate::variable::Module;
use crate::variable::VariableRef;

pub struct Relu {}

impl<T, D> Module<T, D> for Relu
where
    T: NdFloat,
    D: Dimension,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, D>, _y: &'b Array<T, D>) -> Array<T, D> {
        x.mapv(|a| a.max(T::zero()))
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, D>,
        left_ref: &'a VariableRef<T, D>,
        _right_ref: &'a VariableRef<T, D>,
    ) -> [Array<T, D>; 2] {
        let mut grad = grad.clone();
        let ref data = left_ref.borrow().data;

        Zip::from(&mut grad).and(data).for_each(|g, &d| {
            *g = if d.is_sign_positive() { *g } else { T::zero() };
        });

        let zero = Array::<T, D>::zeros(grad.raw_dim());

        [grad, zero]
    }
}

impl<T, D> VariableRef<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    pub fn relu(&mut self) -> VariableRef<T, D> {
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
        let mut x = Variable::new(array!([2.0]));

        assert_eq!(x.relu().borrow().data, array!([2.0]));

        let mut y = Variable::new(array!([-10.0]));

        assert_eq!(y.relu().borrow().data, array!([0.0]));
    }

    #[test]
    fn check_backward_positive() {
        let mut x = Variable::new(array!([2.0]));
        let mut z = x.relu();
        z.backward();
        assert_eq!(x.borrow().get_grad(), array!([1.0]));
    }

    #[test]
    fn check_backward_negative() {
        let mut x = Variable::new(array!([-2.0]));
        let mut z = x.relu();
        z.backward();
        assert_eq!(x.borrow().get_grad(), array!([0.0]));
    }
}
