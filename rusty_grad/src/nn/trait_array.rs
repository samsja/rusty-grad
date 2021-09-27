use ndarray::{Array, Ix1, Ix2};
use std::ops::Add;

struct A {}
struct B {}

impl Add for A {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {}
    }
}

impl Add<B> for A {
    type Output = A;

    fn add(self, other: B) -> Self {
        Self {}
    }
}

pub fn add<T>(x: T, y: T)
where
    T: Add<Output = T>,
{
    x + y;
}

pub fn add2<T, D>(x: T, y: D)
where
    T: Add<D, Output = T>,
{
    x + y;
}

//
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trait_test() {
        let x = Array::<f32, Ix1>::zeros(10);
        let y = Array::<f32, Ix1>::zeros(10);

        add(x, y);

        add(1, 2);

        add(A {}, A {});

        A {} + B {};

        add2(A {}, B {});
    }
}
