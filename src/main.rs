
extern crate num;
extern crate libm;
extern crate ncurses;
extern crate minifb;
use minifb::*;
use libm::*;
mod train;mod data;mod vector;mod escape;mod canvas;mod network;mod window;mod settings;
use window::*;use vector::*;use train::*;use network::*;use escape::*;use canvas::*;use data::*;
use settings::*;
use float_extras::f64::fmod;

fn main() {
	let mut settings = Settings::default_settings();
	//let mut display_error = 0.0;
	let font = loaded_font(1);
	let data_x_range = (settings.data_x_min, settings.data_x_max);
	let data_y_range = (settings.data_y_min, settings.data_y_max);
	let x_range = (settings.x_min,settings.x_max);
	let y_range = (settings.y_min,settings.y_max);
	let mut data = random_regular_datapoints( 3, x_range, y_range);
	let mut f = Network::dense(&settings.nodes_in_layer, settings.random_weight_span);
	let mut g = |x : f64| libm::sin(x);
	//let mut g = random_function();
	let mut display_settings = update_display_settings(&f,&settings);
	let mut learning = true;
	

	let mut graph_canvas = Canvas::new(settings.graph_rows,settings.graph_cols);
	let mut plot_canvas = Canvas::new(settings.plot_rows,settings.plot_cols);
	let mut control_canvas = Canvas::new(settings.control_rows,settings.control_cols);
	let mut canvas = Canvas::new(settings.canvas_rows,settings.canvas_cols);
	let mut canvas_window = Window::new("ffunction", settings.canvas_cols, settings.canvas_rows, 
		WindowOptions::default() ).expect("Unable to open Window");
	control_canvas.add_settings(&display_settings, &font, settings.current_setting);
	let mut data = random_regular_datapoints( settings.datapoints, data_x_range, data_y_range);
	while canvas_window.is_open() && !canvas_window.is_key_down(Key::Escape) {
		let x_range = (settings.x_min,settings.x_max);
		let y_range = (settings.y_min,settings.y_max);
		let data_x_range = (settings.data_x_min, settings.data_x_max);
		let data_y_range = (settings.data_y_min, settings.data_y_max);

		let positions = f.positions(&graph_canvas);

		graph_canvas.clear();plot_canvas.clear();control_canvas.clear();
		let g = model_function(settings.model);
		let f_clo = |x| f.im_fwd(x);
		
		if settings.data_mode {
			plot_canvas.add_data(&data, x_range,y_range, [255,255,0],5);
		} else {
			plot_canvas.add_closure(g, x_range, y_range, [0,0,255], settings.plot_thickness);
		}
			
		graph_canvas.add_network_weights(&f, &positions, settings.weight_limit);
    	graph_canvas.add_network_nodes(&positions,  [255,0,0], 3);
    	graph_canvas.load_cells_from_image();
    	
    	plot_canvas.add_grid(x_range, y_range, [255,255,255]);
		plot_canvas.add_error(settings.display_error, &font, 0,0, [155,155,155]);
		plot_canvas.add_closure(f_clo, x_range, y_range, [255,0,0], settings.plot_thickness);
		plot_canvas.load_cells_from_image();
		
		control_canvas.add_settings(&display_settings, &font, settings.current_setting);
		control_canvas.load_cells_from_image();
		
		
		canvas.inscribe(&graph_canvas, settings.graph_row,settings.graph_col);
		canvas.inscribe(&plot_canvas, settings.plot_row,settings.plot_col);
		canvas.inscribe(&control_canvas, settings.control_row, settings.control_col);
		
		
		
		canvas_window.update_with_buffer(&canvas.cells, settings.canvas_rows, settings.canvas_cols).ok();
		canvas_window.get_keys_pressed(KeyRepeat::Yes).iter().for_each(|key|
			match key {
				Key::C => {
	
				},
				Key::F => 
				{
					data = datapoints_from_function( g, settings.datapoints, x_range);
					settings.data_mode =true;
				},
				Key::X => 
				{
						f.makeover(&[1,11,9,13,9,11,1],2.0);
						display_settings = update_display_settings(&f,&settings);
						let positions = f.positions(&graph_canvas);
						data = random_regular_datapoints( settings.datapoints, data_x_range, data_y_range);
				},
			
				Key::L => learning = ! learning,
				
				Key::B => for _ in 0..settings.manual_training_rounds {
						settings.display_error = train_network_with_data(&mut f, &data);},
				
				Key::D => 
				{
						data = random_regular_datapoints( settings.datapoints, data_x_range, data_y_range);
						settings.data_mode =true;
				},
				Key::W => 
				{
						f.weight = random_vector_in(f.num_nodes*f.num_nodes,(-1.0,1.0));
						f.bias = random_vector_in(f.num_nodes, (-0.00,0.00) );
				}
				Key::Down => 
				{
						settings.current_setting += 1;
						if settings.current_setting > display_settings.len() - 1 {settings.current_setting = 0;}
				},	
				Key::Up => 
				{
						if settings.current_setting >= 1 {
							settings.current_setting -= 1;
						} else {
							settings.current_setting = display_settings.len()-1;
						}
				},
				Key::Right => 
				{
						respond_to_increase(&mut display_settings[settings.current_setting], &mut f, &mut settings);
						let positions = f.positions(&graph_canvas);
						display_settings = update_display_settings(&f,&settings);
						control_canvas.add_settings(&display_settings, &font, settings.current_setting);
				},
				Key::Left => 
				{
						respond_to_decrease(&mut display_settings[settings.current_setting],&mut f, &mut settings);
						let positions = f.positions(&graph_canvas);
						display_settings = update_display_settings(&f,&settings);
						control_canvas.add_settings(&display_settings, &font,settings.current_setting);
				},
				Key::M => settings.data_mode = !settings.data_mode,
				Key::G => 
				{
						settings.model += 1;
						if settings.model >= settings.number_of_models {settings.model = 0};
						let g = model_function(settings.model);
				},
				Key::Q => {
					settings = Settings::default_settings();
					display_settings = update_display_settings(&f,&settings);
				},
								
				_ => (),
			});
		match learning { 
			true => for _ in 0..settings.batch_size {
						settings.display_error = match settings.data_mode {
							true => train_network_with_data(&mut f, &data),
							false => train_network_with_function(&mut f, g, settings.datapoints, x_range),
						}
					},
			false => (),
		}
	}
	pub fn model_function(n : usize) -> fn(f64)->f64 {
		let a = rand::random::<f64>();
		match n {
			0 => |x : f64| 8.0*fmod(4.0*sin(x),2.0),
			1 => |x: f64| 20.0*tanh(fmod(x, sin(x))),
			2 => |x: f64| fmod(x.powf(2.0), 10.0),
			3 => |x: f64| floor(x),
			4 => |x: f64| exp(0.2*sin(x*x*x)*sin(x)),
			5 => |x: f64| if fmod(x,1.0) < 0.5 {-1.0} else {1.0},
			6 => |x :f64| 5.0*fmod(4.0*sin(x),2.0),
			7 => |x: f64| 4.0*floor(x),
			8 => |x: f64| 4.0*sin(x)*floor(x),
			9 => |x: f64| cos(x)*exp(0.2*sin(x*x*x)*sin(x)),
			10 => |x: f64| fmod(x.abs(), 1.0),
			11 => |x: f64| fmod(fmod(x,2.0), 1.0),
			12 => |x: f64| sin(fmod(x,4.0)),
			13 => |x: f64| sin(x),
			14 => |x: f64| exp(x),
			15 => |x: f64| x.powf(3.0)-x.powf(2.0)+1.0,

			_ => |x: f64| x,
		}
	}
}

// pub fn random_function( x : f64, coefficient : &[f64], phase : &[f64], exponent : &[f64] )-> f64 {
// 	let complexity = 10;
// 	let exponent = random_vector_in(complexity,(-5.0,5.0));
// 	let phase = random_vector_in(complexity,(-5.0,5.0));
// 	move |x : f64| {
// 		let mut sum = 0.0;
// 		for i in 0..complexity{
// 			sum += (x - phase[i]).powf(exponent[i]);
// 		}
// 		sum
// 	}
// }
// case 0: g = sin(x);break;
// case 1: g = fmod(x,2)-.5;break;
// case 2: g = .5*x*x-1.5;break;
// case 3: g = 2*exp(-x*x)-.5;break;
// case 4: g = exp(0.2*sin(x*x*x))*sin(x);break;
// case 5: g = fabs(x)-2;break;
// case 6: g = cbrt(x);break;
// case 7: g = floor(x-.2) - 0.50; break;
// case 8: g = 1.5 + sqrt(x)-x;break;
// case 9: g = fmod(x,2); break;
// case 10: if (x > 0) g = 1+exp(-x); else g= .5*x;break;
// case 11: g = 0.2*x*x*x;break;
// case 12: if (fabs(x) > 1) g = -1.5; else g = 1.5;break;
// case 13: g = 0.75 - (abs(floor(x))%2); break;
// case 14: g = 2*sin(4*x);break;
// case 15: g = fabs(1.5*sin(8*x)) - fmod(x,1);break;
// case 16: g = 2-8*fmod(fabs(x),.5);break;
// case 17: g = fmod(fabs(x),1.0);break;
// case 18: g = fmod(fmod(x,2),1);break;
// case 19: h = fmod(fabs(x),1);g =  ((h >0.5)?1-h:h)-.25;break;
// case 20: h = fmod(fabs(4*x),2);g = (h >1)?2-h:h;break;
// case 21: h = fmod(fabs(x),2);g = ((h >1)?2-h:h)-.5;break;
// case 22: b = 4;h = fmod(fabs(x),b); g = ((h>.5*b)?b-h:h)-.25*b;break;






 









