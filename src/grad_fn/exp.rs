use ndarray::{Array, IxDyn, NdFloat};

use crate::variable::GradFn;
use crate::variable::VariableRef;

pub struct Exp {}

impl<T> GradFn<T> for Exp
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, _y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        x.mapv(|a| a.exp())
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        left_ref: &'a VariableRef<T>,
        _right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        let grad = grad.clone();
        let ref data = left_ref.borrow().data;

        let grad = grad * self.forward(data, data);
        let zero = Array::<T, IxDyn>::zeros(grad.raw_dim());

        [grad, zero]
    }
}

impl<T> VariableRef<T>
where
    T: NdFloat,
{
    pub fn exp(&mut self) -> VariableRef<T> {
        let grad_fn = Exp {};
        grad_fn.subscribe(self, self, Box::new(Exp {}))
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

        let res = array!([a.exp()], [b.exp()]);

        assert_eq!(x.exp().borrow().data, res.into_dyn());
    }

    #[test]
    fn check_backward() {
        let a: f32 = 2.0;
        let b: f32 = 1.0;

        let mut x = Variable::new(array!([a], [b]).into_dyn());
        let res = array!([a.exp()], [b.exp()]);

        let mut z = x.exp();
        z.backward();

        assert_eq!(z.borrow().data, res.into_dyn());
        assert_eq!(z.borrow().data, x.borrow().get_grad_f());
    }
}
