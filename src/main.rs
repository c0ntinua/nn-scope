
extern crate num;
extern crate libm;
extern crate ncurses;
extern crate minifb;
use minifb::*;
use libm::*;
mod train;mod data;mod vector;mod escape;mod canvas;mod network;mod window;mod settings;
use window::*;use vector::*;use train::*;use network::*;use escape::*;use canvas::*;use data::*;
use settings::*;

fn main() {
	let mut settings = Settings::default_settings();
	let mut display_error = 0.0;
	let font = loaded_font(1);
	let function_plot_thickness = 2;
	let data_x_range = (settings.data_x_min, settings.data_x_max);
	let data_y_range = (settings.data_y_min, settings.data_y_max);
	let x_range = (settings.x_min,settings.x_max);
	let y_range = (settings.y_min,settings.y_max);
	let mut data = random_regular_datapoints( 3, x_range, y_range);
	let mut f = Network::dense(&settings.nodes_in_layer, settings.random_weight_span);
	let g = libm::sin;
	let g = |x| 8.0*sin(2.0*x);
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
		
		let f_clo = |x| f.im_fwd(x);
		
		if settings.data_mode {
			plot_canvas.add_data(&data, x_range,y_range, [255,255,0],8);
		} else {
			plot_canvas.add_closure(g, x_range, y_range, [0,0,255], function_plot_thickness);
		}
			
		graph_canvas.add_network_weights(&f, &positions, settings.weight_limit);
    	graph_canvas.add_network_nodes(&positions,  [255,0,0], 6);
    	graph_canvas.load_cells_from_image();
    	
    	plot_canvas.add_grid(x_range, y_range, [255,255,255]);
		plot_canvas.add_error(display_error, &font, 0,0, [155,155,155]);
		plot_canvas.add_closure(f_clo, x_range, y_range, [255,0,0], function_plot_thickness);
		plot_canvas.load_cells_from_image();
		
		control_canvas.add_settings(&display_settings, &font, settings.current_setting);
		control_canvas.load_cells_from_image();
		
		canvas.inscribe(&plot_canvas, settings.plot_row,settings.plot_col);
		canvas.inscribe(&graph_canvas, settings.graph_row,settings.graph_col);
		canvas.inscribe(&control_canvas, settings.control_row, settings.control_col);	
		
		canvas_window.update_with_buffer(&canvas.cells, settings.canvas_rows, settings.canvas_cols).ok();
		canvas_window.get_keys_pressed(KeyRepeat::Yes).iter().for_each(|key|
			match key {
				Key::F => 
				{
						f.makeover(&[1,27,31,27,1], 1.0);
						display_settings = update_display_settings(&f,&settings);
						let positions = f.positions(&graph_canvas);		   	
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
						display_error = train_network_with_data(&mut f, &data);},
				
				Key::D => 
				{
						data = random_regular_datapoints( settings.datapoints, data_x_range, data_y_range);
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
								
				_ => (),
			});
		match learning { 
			true => for _ in 0..settings.batch_size {
						display_error = match settings.data_mode {
							true => train_network_with_data(&mut f, &data),
							false => train_network_with_function(&mut f, g, settings.datapoints, x_range),
						}
					},
			false => (),
		}
	}
}







 









