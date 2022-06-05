//use image::{ImageBuffer, RgbImage, GrayImage, Pixel};
use crate::network::*;
use crate::Activation::*;
use crate::DisplaySetting::*;
use rusttype::{Font, Scale};


#[derive(Debug)]
#[derive(Clone)]
#[derive(Copy)]
pub enum Activation {
	Identity,
	Tanh,
	Relu,
	Sin,
	Poly,
}

impl Activation {
	pub fn abbr(&self) -> String {
		match self {
			Activation::Identity => "idn  ".to_string(),
			Activation::Tanh =>     "tnh".to_string(),
			Activation::Relu =>     "rel".to_string(),
			Activation::Sin =>      "sin ".to_string(),
			Activation::Poly =>     "pol".to_string(),
		}
	}
}
		

#[derive(Debug)]
#[derive(Clone)]
pub enum DisplaySetting {
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
	DataXMin(f64),
	DataYMin(f64),
	DataXMax(f64),
	DataYMax(f64),
}

pub struct Settings {
	pub canvas_rows : usize,
	pub canvas_cols : usize,
	pub control_rows : usize,
	pub control_cols : usize,
	pub graph_rows : usize,
	pub graph_cols : usize,
	pub plot_rows : usize,
	pub plot_cols : usize,
	pub control_row : usize,
	pub control_col : usize,
	pub graph_row : usize,
	pub graph_col : usize,
	pub plot_row : usize,
	pub plot_col : usize,
	pub current_setting : usize,
	pub nodes_in_layer : Vec<usize>, 
	pub x_min : f64,
	pub x_max : f64,
	pub y_min : f64,
	pub y_max : f64,
	pub data_x_min : f64,
	pub data_x_max : f64,
	pub data_y_min : f64,
	pub data_y_max : f64,
	pub datapoints : usize,
	pub batch_size : usize,
	pub manual_training_rounds : usize,
	pub random_weight_span : f64,
	pub weight_limit : f64,
	pub f_color : [u8;3],
	pub data_color : [u8;3],
	pub data_mode : bool,
	pub learning : bool,
	pub plot_thickness : usize,
	pub model : usize,
	pub number_of_models : usize,
	pub display_error : f64,
}

impl Settings {
	pub fn default_settings()->Settings {
		Settings {
			canvas_rows : 1100,
			canvas_cols : 1100,
			control_rows : 1100,
			control_cols : 190,
			graph_rows : 500,
			graph_cols : 1000,
			plot_rows : 580,
			plot_cols : 910,
			control_row : 0,
			control_col : 0,
			graph_row : 10,
			graph_col : 130,
			plot_row : 505,
			plot_col : 170,
			current_setting : 0,
			nodes_in_layer : vec![1,9,9,9,9,9,1],
			x_min : -10.0,
			x_max : 10.0,
			y_min : -5.0,
			y_max :  5.0,
			data_x_min : -10.0,
			data_x_max : 10.0,
			data_y_min : -5.0,
			data_y_max : 5.0,
			datapoints : 7,
			batch_size : 100,
			manual_training_rounds : 500,
			random_weight_span : 1.0,
			weight_limit : 2.0,
			f_color : [255,0,0],
			data_color : [0,255,255],
			data_mode : true,
			learning : true,
			plot_thickness : 12,
			model : 0,
			number_of_models : 20,
			display_error : 0.0,

		
		}
	}
}

pub fn loaded_font(code : usize) -> Font <'static> {
	let font_code =  match code {
		0 => Vec::from(include_bytes!( "cc.ttf") as &[u8]),
		1 => Vec::from(include_bytes!( "scp.ttf") as &[u8]),
		2 => Vec::from(include_bytes!( "german.ttf") as &[u8]),
		3 => Vec::from(include_bytes!( "sg.ttf") as &[u8]),
		4 => Vec::from(include_bytes!( "lr.ttf") as &[u8]),
		
		_ => Vec::from(include_bytes!( "mt.ttf") as &[u8]),
	};
	Font::try_from_vec(font_code).unwrap()
}


pub fn update_display_settings(network : &Network, settings : &Settings) -> Vec<DisplaySetting> {
	let mut display_settings = vec![];
	display_settings.push(DisplaySetting::Rate(network.rate[0]));
	display_settings.push(DisplaySetting::NumLayers(network.num_layers));
	display_settings.push(DisplaySetting::Datapoints(settings.datapoints));
	display_settings.push(DisplaySetting::BatchSize(settings.batch_size));

	for i in 0..network.num_layers {
		display_settings.push(DisplaySetting::NodesInLayer{ num_nodes : network.nodes_in_layer[i], layer : i });
		
	}
	for i in 0..network.num_layers {
		display_settings.push(DisplaySetting::ActivationOfLayer{ act : network.act[i], layer : i });
	}
	display_settings.push(DisplaySetting::WeightLimit(settings.weight_limit));
	display_settings.push(DisplaySetting::XMin(settings.x_min));
	display_settings.push(DisplaySetting::XMax(settings.x_max));
	display_settings.push(DisplaySetting::YMin(settings.y_min));
	display_settings.push(DisplaySetting::YMax(settings.y_max));
	display_settings.push(DisplaySetting::DataXMin(settings.data_x_min));
	display_settings.push(DisplaySetting::DataXMax(settings.data_x_max));
	display_settings.push(DisplaySetting::DataYMin(settings.data_y_min));
	display_settings.push(DisplaySetting::DataYMax(settings.data_y_max));
	display_settings
}

pub fn respond_to_increase(display_setting : &mut DisplaySetting, f : &mut Network, settings : &mut Settings) {
	match display_setting {
		&mut DisplaySetting::NumLayers(n) => {
			if n < 21 {
				*display_setting = DisplaySetting::NumLayers(n+1);
				f.nodes_in_layer.insert(n-1,9);
				let next = f.nodes_in_layer.clone();
				f.makeover(&next, settings.random_weight_span);
			}
		},
		&mut DisplaySetting::Rate(r) => {  
			*display_setting = DisplaySetting::Rate(r+0.000001);
			f.rate = vec![r+0.000001;f.num_layers];
		},
		&mut DisplaySetting::WeightLimit(l) => {  
			*display_setting = WeightLimit(l+1.0);
			settings.weight_limit = l + 1.0;
		},
		&mut DisplaySetting::BatchSize(s) => {  
			*display_setting = BatchSize(s+1);
			settings.batch_size= s + 1;
		},
		&mut DisplaySetting::Datapoints(s) => {
			if settings.datapoints < 100 {
				*display_setting = Datapoints(s+1);
				settings.datapoints= s +1;
			}
		},
		&mut DisplaySetting::NodesInLayer{num_nodes :n , layer : l} => {
			if f.nodes_in_layer[l] <= 57 && l != 0 && l != f.num_layers - 1 {
				*display_setting = NodesInLayer { num_nodes : n+2 , layer :l };
				f.nodes_in_layer[l] = n +2;
				let next = f.nodes_in_layer.clone();
				f.makeover(&next, settings.random_weight_span);
			}
		},
		&mut DisplaySetting::ActivationOfLayer{act : a , layer : l} => {
			let next = match a {
				Identity => Relu,
				Relu => Tanh,
				Tanh =>Poly,
				Poly => Sin,
				Sin => Identity,
			};
			*display_setting = ActivationOfLayer{act : next , layer : l};
			f.act[l] = next;
		},
		&mut DisplaySetting::XMax(l) => {  
			*display_setting = XMax(l+1.0);
			settings.x_max = l + 1.0;
		},
		&mut DisplaySetting::XMin(l) => {
			if settings.x_min + 1.0 < settings.x_max {  
				*display_setting = XMin(l+1.0);
				settings.x_min = l + 1.0;
			}
		},
		&mut DisplaySetting::DataXMin(l) => {
			if settings.data_x_min + 1.0 < settings.data_x_max {  
				*display_setting = DataXMin(l+1.0);
				settings.data_x_min = l + 1.0;
			}
		},
		&mut DisplaySetting::YMax(l) => {  
			*display_setting = YMax(l+1.0);
			settings.y_max = l + 1.0;
		},
		&mut DisplaySetting::DataYMax(l) => {  
			*display_setting = DataYMax(l+1.0);
			settings.data_y_max = l + 1.0;
		},
		&mut DisplaySetting::DataXMax(l) => {  
			*display_setting = DataXMax(l+1.0);
			settings.data_x_max = l + 1.0;
		},
		&mut DisplaySetting::YMin(l) => {
			if settings.y_min +1.0 < settings.y_max {  
				*display_setting = YMin(l+1.0);
				settings.y_min = l + 1.0;
			}
		},
		&mut DisplaySetting::DataYMin(l) => {
			if settings.data_y_min + 1.0 < settings.data_y_max {  
				*display_setting = DataYMin(l+1.0);
				settings.data_y_min = l + 1.0;
			}
		},
		_ => (),
	}											
}
pub fn respond_to_decrease(display_setting : &mut DisplaySetting, f : &mut Network, settings : &mut Settings) {
	match display_setting {
		&mut DisplaySetting::NumLayers(n) => {
			if n > 3 {
				*display_setting = NumLayers(n-1);
				f.nodes_in_layer.remove(n-2);
				let next = f.nodes_in_layer.clone();
				f.makeover(&next, settings.random_weight_span);
			}
		},
		&mut DisplaySetting::Rate(r) => {  
			*display_setting = Rate(r-0.000001);
			f.rate = vec![r-0.000001;f.num_layers];
		},
		&mut DisplaySetting::WeightLimit(l) => {
			if settings.weight_limit >= 2.0 { 
				*display_setting = WeightLimit(l-1.0);
				settings.weight_limit = l - 1.0;
			}
		},
		&mut DisplaySetting::BatchSize(s) => {
			if settings.batch_size >= 2 {
				*display_setting = BatchSize(s-1);
				settings.batch_size= s - 1;
			}
		},
		&mut DisplaySetting::Datapoints(s) => {
			if settings.datapoints >= 2 {
				*display_setting = Datapoints(s-1);
				settings.datapoints= s - 1;
			}
		},
		&mut DisplaySetting::NodesInLayer{num_nodes :n , layer : l} => {
			if f.nodes_in_layer[l] >= 3 && l != 0 && l != f.num_layers - 1 {
				*display_setting = DisplaySetting::NodesInLayer{ num_nodes : n- 2 , layer :l };
				f.nodes_in_layer[l] = n  - 2;
				let next = f.nodes_in_layer.clone();
				f.makeover(&next, settings.random_weight_span);
			}
		},
		&mut DisplaySetting::ActivationOfLayer{act : a , layer : l} => {
			let next = match a {
				Relu =>Identity,
				Tanh => Relu,
				Poly => Tanh,
				Sin => Poly,
				Identity => Sin,
			};
			*display_setting = ActivationOfLayer{act : next , layer : l};
			f.act[l] = next;
		},
		&mut DisplaySetting::XMin(l) => {  
			*display_setting = XMin(l-1.0);
			settings.x_min = l - 1.0;
		},
		&mut DisplaySetting::DataXMin(l) => {  
			*display_setting = DataXMin(l-1.0);
			settings.data_x_min = l - 1.0;
		},
		&mut DisplaySetting::XMax(l) => {
			if settings.x_min +1.0 < settings.x_max {  
				*display_setting = XMax(l-1.0);
				settings.x_max = l - 1.0;
			}
		},
		&mut DisplaySetting::YMin(l) => {  
			*display_setting = YMin(l-1.0);
			settings.y_min = l - 1.0;
		},
		&mut DisplaySetting::DataYMin(l) => {  
			*display_setting = DataYMin(l-1.0);
			settings.data_y_min = l - 1.0;
		},
		&mut DisplaySetting::YMax(l) => {
			if settings.y_min +1.0 < settings.y_max {  
				*display_setting = YMax(l-1.0);
				settings.y_max = l - 1.0;
			}
		},
		&mut DisplaySetting::DataYMax(l) => {
			if settings.data_y_min +1.0 < settings.data_y_max {  
				*display_setting = DataYMax(l-1.0);
				settings.data_y_max = l - 1.0;
			}
		},
		&mut DisplaySetting::DataXMax(l) => {
			if settings.data_x_min + 1.0 < settings.data_x_max {  
				*display_setting = DataXMax(l-1.0);
				settings.data_x_max = l - 1.0;
			}
		},
		_ => (),
		
	}
}
