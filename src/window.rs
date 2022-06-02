use minifb::*;

pub fn system_window(rows : usize, cols : usize)  -> Window {
	Window::new("", cols, rows, WindowOptions::default() ).expect("Unable to open Window")
}

	