use ndarray::{Array, Dimension, NdFloat};

use crate::variable::VariableRef;

pub fn max<T, D>(x: &Array<T, D>) -> T
where
    T: NdFloat,
    D: Dimension,
{
    let mut max = T::zero();

    for val in x.iter() {
        if *val > max {
            max = *val;
        }
    }
    max
}

impl<T> VariableRef<T>
where
    T: NdFloat,
{
    pub fn softmax(&mut self) -> VariableRef<T> {
        let mut copy = self.identity();
        copy.borrow_mut().data -= max(&self.borrow().data);
        let mut exp = copy.exp();

        let exp_sum = exp.sum();

        exp / exp_sum
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
