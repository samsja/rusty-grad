use crate::variable::VariableRef;

use ndarray::NdFloat;

pub trait Module<T: NdFloat> {
    fn params(&self) -> Vec<VariableRef<T>>;

    fn f(&mut self, input: &VariableRef<T>) -> VariableRef<T>;

    fn zero_grad(&mut self) {
        for p in self.params().iter_mut() {
            p.borrow_mut().zero_grad();
        }
    }
}
