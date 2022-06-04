
use crate::network::*;

pub fn train_network_with_data(f : &mut Network, data : &[(f64,f64)]) -> f64 {
	let mut abs_error =vec![];
	for datum in data.iter() {
		let guess = f.fwd(datum.0);
		let error = guess - datum.1;
		f.rtr(error);
		abs_error.push(error.abs());
	}
	abs_error.into_iter().fold(0.0, |sum, next| sum + next )/(data.len() as f64)
}

pub fn train_network_with_function(f : &mut Network, g : fn(f64)->f64, num_examples : usize, x_range : (f64, f64)) -> f64 {
	let mut abs_error =vec![0.0;num_examples];
	for _ in 0..num_examples {
		let x = x_range.0 + rand::random::<f64>() *( x_range.1 - x_range.0);
		let guess = f.fwd(x);
		let target = g(x);
		let error = guess - target;
		f.rtr(error);
		abs_error.push(error.abs());
	}
	abs_error.into_iter().fold(0.0, |sum, next| sum + next )/(num_examples as f64)
}