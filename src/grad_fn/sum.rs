use ndarray::{Array, Ix1, IxDyn, NdFloat};

use crate::variable::GradFn;
use crate::variable::VariableRef;

pub struct Sum {}

impl<T> GradFn<T> for Sum
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, _y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        Array::<T, Ix1>::from_vec(vec![x.sum()]).into_dyn()
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        left_ref: &'a VariableRef<T>,
        _right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        let grad = grad.clone();
        let ref data = left_ref.borrow().data;

        let grad = grad * Array::<T, IxDyn>::ones(data.raw_dim());
        let zero = Array::<T, IxDyn>::zeros(grad.raw_dim());

        [grad, zero]
    }
}

impl<T> VariableRef<T>
where
    T: NdFloat,
{
    pub fn sum(&mut self) -> VariableRef<T> {
        let grad_fn = Sum {};
        grad_fn.subscribe(self, self, Box::new(Sum {}))
    }
}

#[cfg(test)]
mod tests {

    use crate::variable::Variable;
    use ndarray::array;

    #[test]
    fn check_method() {
        let a: f32 = 2.0;
        let b: f32 = 1.0;

        let mut x = Variable::new(array!([a], [b]).into_dyn());

        let res = array!(a + b);

        assert_eq!(x.sum().borrow().data, res.into_dyn());
    }

    #[test]
    fn check_backward() {
        let a: f32 = 2.0;
        let b: f32 = 1.0;

        let mut x = Variable::new(array!([a], [b]).into_dyn());

        let res = array!([1.], [1.]);

        let mut z = x.sum();
        z.backward();

        assert_eq!(x.borrow().get_grad_f(), res.into_dyn());
    }
}
