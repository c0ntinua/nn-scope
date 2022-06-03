//use image::{ImageBuffer, RgbImage, GrayImage, Pixel};
use crate::network::*;
use rusttype::{Font, Scale};

#[derive(Debug)]
#[derive(Clone)]
#[derive(Copy)]
pub enum Activation {
	Identity,
	Tanh,
	Relu,
	Sin,
}

impl Activation {
	pub fn abbr(&self) -> String {
		match self {
			Activation::Identity => "id  ".to_string(),
			Activation::Tanh =>     "tanh".to_string(),
			Activation::Relu =>     "relu".to_string(),
			Activation::Sin =>      "sin ".to_string(),
		}
	}
}
		

#[derive(Debug)]
#[derive(Clone)]
pub enum Setting {
	CurrentSetting(usize),
	NumLayers(usize),
	BatchSize(usize),
	WeightLimit(f64),
	NodesInLayer { num_nodes : usize, layer : usize },
	ActivationOfLayer{ act : Activation, layer : usize},
	RateOfLayer { rate : f64, layer : usize},
	Rate(f64),
	Datapoints(usize),
	XMin(f64),
	XMax(f64),
	YMin(f64),
	YMax(f64),
}



pub fn loaded_font(code : usize) -> Font <'static> {
		let font_code =  match code {
			0 => Vec::from(include_bytes!( "cc.ttf") as &[u8]),
// 			1 => Vec::from(include_bytes!( "scp.ttf") as &[u8]),
// 			2 => Vec::from(include_bytes!( "german.ttf") as &[u8]),
// 			3 => Vec::from(include_bytes!( "sg.ttf") as &[u8]),
// 			4 => Vec::from(include_bytes!( "lr.ttf") as &[u8]),
// 			
			_ => Vec::from(include_bytes!( "mt.ttf") as &[u8]),
		};
		Font::try_from_vec(font_code).unwrap()
}


pub fn settings_from_network(network : &Network) -> Vec<Setting> {
	let mut settings = vec![];
	settings.push(Setting::Rate(network.rate[0]));
	settings.push(Setting::NumLayers(network.num_layers));
	for i in 0..network.num_layers {
		settings.push(Setting::NodesInLayer{ num_nodes : network.nodes_in_layer[i], layer : i });
		
	}
	for i in 0..network.num_layers {
		settings.push(Setting::ActivationOfLayer{ act : network.act[i], layer : i });
	}
// 	for i in 0..network.num_layers {
// 		settings.push(Setting::RateOfLayer{ rate : network.rate[i], layer : i });
// 	}
	settings.push(Setting::WeightLimit(network.weight_limit));
	settings.push(Setting::BatchSize(network.batch_size));
	settings.push(Setting::Datapoints(network.datapoints));
	settings.push(Setting::XMin(network.x_min));
	settings.push(Setting::XMax(network.x_max));
	settings.push(Setting::YMin(network.y_min));
	settings.push(Setting::YMax(network.y_max));
	settings
}


	


// pub fn settings() -> Vec<Setting> {
// 	let max_layers = 7;
// 	let max_nodes_in_layer = 31; 
// 	let mut settings = vec![];
// 	settings.push(Setting::NumLayers(network.num_layers));
// 	for i in 0..network.max_layers {
// 		settings.push(Setting::RateOfLayer{ rate : network.rate[i], layer : i });
// 	}
// 	for i in 0..network.max_layers {
// 		settings.push(Setting::NodesInLayer{ num_nodes : network.nodes_in_layer[i], layer : i });
// 	}
// 	settings.push(Setting::BatchSize(100));
// 	settings
// }
