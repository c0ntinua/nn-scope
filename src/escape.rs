#![allow(dead_code, unused,non_snake_case)]
pub fn go(x: usize, y : usize) {
	print!("\u{1B}[{};{}H",y+1,x);
}

pub fn cls() {
	print!("\u{1B}[2J");
}
pub fn activate_bold_text() {
	print!("\u{1B}[1m");
}
pub fn activate_reverse_text() {
	print!("\u{1B}[7m");
}
pub fn deactivate_bold_text() {
	print!("\u{1B}[22m");
}
pub fn deactivate_reverse_text() {
	print!("\u{1B}[27m");
}

pub fn set_rgb(r : u8, g : u8, b : u8) {
	print!("\u{1B}[38;2;{};{};{}m",r,g,b);
}




