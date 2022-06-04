
use crate::vector::*;
use crate::network::*;

pub fn train_network(f : &mut Network, F : fn(f64) -> f64, n : usize, x_range : (f64, f64)) {
	for i in 0..n {
		let x = x_range.0 + rand::random::<f64>() *( x_range.1 - x_range.0);
		let guess = f.fwd(x);
		let target = F(x);
		f.rtr(guess -target);
	}
	f.clean();
}

pub fn train_network_with_data(f : &mut Network, data : &[(f64,f64)]) {
	for round in 0..f.batch_size {
		let num_data = data.len(); 
		for _ in 0..num_data {
			let i = rand::random::<usize>()%num_data;
			let guess = f.fwd(data[i].0); 
			f.rtr(guess-data[i].1);
		}
	}
}
pub fn train_network_with_data1(f : &mut Network, data : &[(f64,f64)]) -> f64 {
	let mut abs_error =vec![];
	for datum in data.iter() {
		let guess = f.fwd(datum.0);
		let error = guess - datum.1;
		f.rtr(error);
		abs_error.push(error.abs());
	}
	abs_error.into_iter().fold(0.0, |sum, next| sum + next )/(data.len() as f64)
}

pub fn train_network_with_closure<F>(f : &mut Network, g : F , x_range : (f64, f64)) 
	where F : Fn(f64) -> f64 {
	for i in 0..f.batch_size {
		let x = x_range.0 + rand::random::<f64>() *( x_range.1 - x_range.0);
		let guess = f.fwd(x);
		let target = g(x);
		f.rtr(guess -target);
	}
	f.clean();
}
pub fn train_network_with_function1(f : &mut Network, g : fn(f64)->f64, x_range : (f64, f64)) -> f64 {
	let mut abs_error =vec![0.0;f.batch_size];
	for _ in 0..f.datapoints {
		let x = x_range.0 + rand::random::<f64>() *( x_range.1 - x_range.0);
		let guess = f.fwd(x);
		let target = g(x);
		let error = guess - target;
		f.rtr(error);
		abs_error.push(error.abs());
	}
	abs_error.into_iter().fold(0.0, |sum, next| sum + next )/(f.batch_size as f64)
}