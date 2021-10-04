use ndarray::NdFloat;

use crate::variable::{Variable, VariableRef};

impl<T> VariableRef<T>
where
    T: NdFloat,
{
    pub fn softmax(&mut self) -> VariableRef<T> {
        let exp = self.exp();

        let exp_clone = Variable::new_no_retain_grad(exp.borrow().data.clone()).sum();

        exp / exp_clone
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

        let sum_exp = a.exp() + b.exp();
        let res = array!([a.exp() / sum_exp], [b.exp() / sum_exp]);

        assert_eq!(x.softmax().borrow().data, res.into_dyn());
    }
}
