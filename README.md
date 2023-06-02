# King-County-Homebuyer-Help
Flatiron Project 2
King County 
Homebuyer Help
Daniel Cotler, Justin Lapidus, Eitan Westrich
 
## Overview
We have been tasked with analyzing the data on houses in King County, Washington. Our goal is to help prospective buyers understand the housing market better. We did this by creating predictions about the sale price of houses, in King County, based on different features. After creating a linear regression model and constructing a predictive model, we determined that the features with the largest weight, and the strongest correlation to price were square feet of living space and the number of bathrooms.
## Business Problem
King County is the most populous county in the state of Washington and the thirteenth most popular in the United States of America. In Kings County, the largest populated city is Seattle with a total of over 700,000 people. Recently, Seattle has made itself a home to new innovations. Starbucks, Costco, and Microsoft are just a few of the titans of the industry located within. Additionally, the largest employer of the county is Amazon with over 75000 employees in Washington alone. The new influx of business created a pipeline for new prospective workers. These workers are searching for employment and housing options. However, the process of resettling is more complex than expected. Questions begin to compound and answers are seldom found. 


## Data
We utilized a few different data sources for our model so that we could obtain a comprehensive and accurate prediction of home prices.
King County House Data:
A data source that provided the following information; date, number of floors, waterfront property, greenbelt property, nuisance, view, condition, long, lat, grade, heat_source, sewer_system, sqft_garage, sqft_patio, yr_built,y r_renovated, address, price, bedrooms, bathrooms, sqft_living, sqft_lot, sqft_above, and sqft_basement. 

## Methods
The majority of the dataset pertained to houses sold in King County, Washington, but there were several outliers that needed to be addressed. After focusing on just King County we performed a heatmap to show any correlations between the different features and prices. 

<img width="477" alt="Screen Shot 2023-06-02 at 1 42 48 PM" src="https://github.com/ewestrich/King-County-House-Sales/assets/130884190/b20f7524-c7a1-491d-a1e6-9cf97787b096">


 From the following graphic we were able to identify number of bathrooms and square feet of living has the greatest correlation to the price of the house. The number of bathrooms has a .49 correlation while Sqft_living has a .62 correlation on our heatmap. 
<img width="553" alt="Screen Shot 2023-06-02 at 1 42 58 PM" src="https://github.com/ewestrich/King-County-House-Sales/assets/130884190/3dd710a7-aec3-44ac-9830-e0342fe641e0">


We further narrowed the scope of our research to focus on the technology hub in downtown Seattle. In particular, we identified Amazon as the largest employer and point of interest to wanna-be buyers. The range was focused to be a 3-mile radius around Amazon Center. What was discovered was a strong correlation between the price of houses with the sqft_living. We were able to form a regression line to model the relationship. 

After focusing on the correlation between square feet of living and price, we turned our attention to the feature that had the second heaviest weight, the number of bathrooms.  What was observed was that the number of bathrooms has a direct correlation with the price of the house. However, the data is slightly skewed by the limited data points with houses beyond six bathrooms. 

<img width="607" alt="Screen Shot 2023-06-02 at 1 56 01 PM" src="https://github.com/ewestrich/King-County-House-Sales/assets/130884190/aaba048d-0c76-4df2-8d85-e54c1e8a69d8">



## Results
We fit the following data into a multivariable linear regression model to establish a fairly accurate price guage, based upon features represented within the data. We achieved an R-square score of .66 indicating that our predicted value will fall with 66 percent accuracy.



## Analysis of Model
Our final model includes aspects of all numerical, ordinal, and nominal categories from our data frame and the influence of each category on our predicted price. About 65% of the variability observed in the sale price is explained by the regression model. Our model is a predictor of house prices based on certain features in King County. Some coefficients to note that have an increase in house sale price include bathrooms, sqft_living, sqft_above, sqft_basement, sqft_patio, grade, condition, and view. The following visual further explains the effectiveness of predicting price by fitting a regression line between the actual and predicted price value. 

<img width="225" alt="Screen Shot 2023-06-02 at 1 55 00 PM" src="https://github.com/ewestrich/King-County-House-Sales/assets/130884190/1da8cd5c-09fa-4905-a303-757cef693b63">

## Further Exploration Through Feature Tooling 

In order to create a real-world application of our data, our team devised a tool to be used by prospective buyers. Buyers can gauge potential housing prices based on the features of the home. Additionally, users can see if the sale price of a home is valued accurately, or not. The tool filters housing data using the following features; zip code, square footage of the house, number of bathrooms, number of floors, number of bedrooms, and more. 

## Recommendations

Our recommendations are as follows:
increase the square footage of living space by adding housing extension
Purchase a less expensive home, and then add bathrooms 
Use our tool to help gauge housing prices before making a purchase
