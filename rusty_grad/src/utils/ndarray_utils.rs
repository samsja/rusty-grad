use ndarray::{stack, Array, ArrayView, Axis, Dimension, NdFloat, RemoveAxis, ShapeError};

pub fn repeat<T, D>(ax: Axis, x: &Array<T, D>, n: usize) -> Result<Array<T, D::Larger>, ShapeError>
where
    T: NdFloat,
    D: Dimension,
    D::Larger: RemoveAxis,
{
    let l: Vec<ArrayView<T, D>> = (0..n).map(|_| x.view()).collect();
    stack(ax, &l)
}

// #[cfg(test)]
// mod tests {
//
//     use ndarray::{array,Axis};
//     use super::repeat;
//
//     #[test]
//     fn repeat_vect() {
//         let x = array!([1.0],[2.0]);
//         assert_eq!(repeat(Axis(1),&x,2).unwrap(),array!([1.0,1.0],[2.0,2.0]) );
//     }
/* } */
