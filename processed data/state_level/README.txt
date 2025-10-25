State level data follows the same idea as county level

 - The adjacency matrix is a linear combination of highway, airport, and border weights
	- These files are the three weights files

 - The cases/covariates data is daily counts in the daily covariates file, and a rolling average in the rolled covariates file
	- I recommend using the rolling average data

 - The three "component" folders have the unscaled and constituent component data that go into the weights
	- The point of these files are so that you can scale the weights however you want (row-norm, column-norm or something else)
	- You can also adjust the relative weights on components if needed (ex. more weight on travel time for highway weights)
	- The weights are a simple gravity model with equal scale on all components
		- The exact formulas in the state level data processing file
