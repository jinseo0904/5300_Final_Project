#install.packages("gt")
#install.packages("webshot2")
webshot2::install_phantomjs()  # Required for HTML rendering

# Load the knitr package
library(knitr)
library(kableExtra)
library(magrittr)
library(tidyr)
library(dplyr)
library(gt)
library(webshot2)
library(tibble)

# List all CSV files in the directory
input_dir <- "/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Jinseo/results"
csv_files <- list.files(input_dir, pattern = "\\.csv$", full.names = TRUE)

# Loop through each CSV file
for (csv_file in csv_files) {
  print(csv_file)
  file_path <- csv_file
  
  # Extract the file name without the extension
  file_name <- tools::file_path_sans_ext(basename(file_path))
  subject <- substr(file_name, start = 1, stop = nchar(file_name) - 13)
  subject_new <- paste0("Subject: ", subject)
  
  # Load the CSV file into a data frame
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  # Combine correlation, p_value, direction, and method into a single string
  # Add stars based on p-value and update matrix entry
  data <- data %>%
    mutate(
      # Determine symbol for the correlation method
      symbol = ifelse(method == "Pearson", "r", ifelse(method == "Spearman", "ρ", "")),
      
      # Add bold stars based on p-value thresholds
      stars = ifelse(p_value < 0.01, "**", ifelse(p_value >= 0.01 & p_value <= 0.05, "*", "")),
      
      # Combine into matrix entry
      matrix_entry = paste0(
        symbol, ": ", round(correlation, 3), stars
      )
    )
  
  # Create the matrix
  matrix_result <- data %>%
    select(x_var, y_var, matrix_entry) %>%
    pivot_wider(names_from = x_var, values_from = matrix_entry) %>%
    column_to_rownames(var = "y_var")
  
  # Specify the save path for the HTML and PNG files
  html_file <- "/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Jinseo/plots/table_output.html"
  png_file <- paste0("/Users/jinseokim/Downloads/NEU/HINF 5300/Final_Project/5300_Final_Project/Jinseo/plots/", subject, ".png")
  
  # Convert matrix to a data frame for gt
  matrix_result_df <- as.data.frame(matrix_result)
  
  # Calculate adaptive vwidth based on number of columns
  num_columns <- ncol(matrix_result_df) - 1  # Exclude the row names column
  vwidth <- 400 + (num_columns * 200)  # 400 for y-axis + 200 per column
  
  # Format stars and apply bold style directly
  table <- matrix_result_df %>%
    rownames_to_column(var = "y_var") %>%
    gt(rowname_col = "y_var") %>%
    tab_header(
      title = subject_new
    ) %>%
    opt_row_striping() %>%
    tab_style(
      style = cell_text(weight = "bold"),  # Boldface y-axis labels
      locations = cells_stub()            # Applies style to the row names (y-axis)
    ) %>%
    tab_style(
      style = cell_text(weight = "bold"),  # Boldface x-axis labels
      locations = cells_column_labels()   # Applies style to the column labels (x-axis)
    ) %>%
    # Change background color of the title
    tab_style(
      style = cell_fill(color = "navy"),  # Customize the color as needed
      locations = cells_title(groups = "title")
    ) %>%
    tab_style(
      style = cell_text(color = "white"),  # Customize the color as needed
      locations = cells_title(groups = "title")
    ) %>%
    # Hide NA values
    fmt_missing(
      columns = everything(),
      missing_text = ""  # Replace NA with blank
    ) %>%
    # Increase horizontal margin between columns
    cols_width(
      y_var ~ px(180),
      everything() ~ px(200)
    )
  
  # Save the table as a PNG with adaptive vwidth
  gtsave(table, png_file, vwidth = vwidth, vheight = 1000)
  
  
}

