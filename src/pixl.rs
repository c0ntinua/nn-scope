pub struct Pixl {
	pub height  : usize,
	pub width : usize,
	pub buffer_rows : usize,
	pub buffer_cols : usize,
	pub rows : usize,
	pub cols : usize,
	pub heights : Vec<usize>,
	pub height_index : usize,
	pub widths : Vec<usize>,
	pub width_index : usize,
}

impl Pixl {
	pub fn default_pixl(height : usize, width : usize, buffer_rows : usize, buffer_cols : usize) -> Pixl {
		Pixl {
			height,
			width,
			buffer_rows,
			buffer_cols,
			rows : buffer_rows/height,
			cols : buffer_cols/width,
			heights : vec![1,2,5,10,20,50],
			height_index : 0,
			widths :  vec![1,2,5,10,20,50],
			width_index : 0,	
		}
	}
	pub fn increase_height(&mut self) {
		if self.height_index < self.heights.len() - 1 {
			self.height_index += 1;	
		} else {
			self.height_index = 0;
		}
		self.height = self.heights[self.height_index];
		self.rows = self.buffer_rows/self.height;
	}
	pub fn increase_width(&mut self) {
		if self.width_index < self.widths.len() - 1 {
			self.width_index += 1;	
		} else {
			self.width_index = 0;
		}
		self.width = self.widths[self.width_index];
		self.cols = self.buffer_cols/self.width;
	}
			
	
	
}
		