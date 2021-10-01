use crate::variable::VariableRef;

use ndarray::NdFloat;

pub trait Optim {
    fn step(&mut self);
}

pub struct SGD<T: NdFloat> {
    pub params: Vec<VariableRef<T>>,
    pub lr: T,
}

impl<T: NdFloat> SGD<T> {
    pub fn new<'a>(params: Vec<VariableRef<T>>, lr: T) -> Result<SGD<T>, &'a str> {
        let mut no_retain = false;
        for p in params.iter() {
            if !p.borrow().is_grad_retain() {
                no_retain = false;
            }
        }

        if no_retain {
            Err("One of the parameters does not retrain his grad no optim possible for it")
        } else {
            Ok(SGD { params, lr })
        }
    }
}

impl<T: NdFloat> Optim for SGD<T> {
    fn step(&mut self) {
        for p in self.params.iter_mut() {
            let mut param_var = p.borrow_mut();
            param_var.data = param_var.data.clone() - param_var.get_grad().unwrap() * self.lr;
        }
    }
}
