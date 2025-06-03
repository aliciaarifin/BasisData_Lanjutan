import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.stats as sc
import seaborn as sns

st.title("Big Data Visualisation")
st.markdown("Alicia Arifin, Basis Data Lanjutan")

st.header("Flowchart")

st.image("Finals_BDL_Streamlit/FitRegression.png")

st.markdown("First, the data will ke imported to python. One of those variables will be chosen to be dependent variables. Inthis case study, that variabels is exam score. For best modeling, numeric variables will be chosen using correlation and categoric variables will be using anova. The chosen variabels using correlation when the correlation with exam score's more than 0,5 (strong correlation). The chosen variabels using anova when the variables have a significant difference from groups. After knowing which's which, the best variables are going to be analyze using regression. Regression that are going to be used are linear regression, dummy regression, and non-linear distributions regression.")


st.header("Data: Students Exam Performance")


st.markdown("The data was from [kaggle](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance) about students performance. The data filled with 1000 rows and 16 columns. The columns are student id, age, gender, study hours per day, social media hours, netflix hours, part time job, attandance percentage, sleep hours, diet quality, excercise frequency, parental educational level, internet quality, mental health rating, extracurricular participation and exam score. From these variables, the dependent variable or target variables is exam score. From this data, we will try to see what variables are maybe best for predicting exam score. To show the data, i use the code right below this text using pandas. Because the data have 1000 rows, i only show the first 10 rows. Later, student id will be removed because student id are index of the students.")


students = pd.read_csv("student_habits_performance.csv")

st.code(
'''
students = pd.read_csv("student_habits_performance.csv")
students.head(10)
'''
)

st.table(students.head(10))


st.code('''
students.info
''')
st.text(students.info)

st.markdown("The data have 1000 rows and 16 columns. From this data, later will found what's the best way to predict exam score. Maybe will using variour regresssion to predict exam score. Choosing independent variables will be separated between categorical and numerical variables. I will be using correlation for numerical varibles, and anova (analysis of variance) for categorical variables.")


st.code(
'''
students.describe()
'''
)
st.table(students.describe())


st.markdown('''
From all of those nueric variables, descriptive analysis is shown below:
- Age: The average age is about 20.5 years old. The ages range from 17 to 24 years.
- Study Hours: On average, students study around 3.5 hours per day. Some don't study at all (0 hours), while others study up to 8.3 hours.
- Social Media Hours: Students spend an average of 2.5 hours on social media daily. This varies from 0 to 7.2 hours.
- Netflix Hours: The average time spent on Netflix is about 1.8 hours per day, with a maximum of 5.4 hours.
- Attendance Percentage: Students, on average, attend 84.1% of their classes. The lowest attendance is 56%, and some have 100% attendance.
- Sleep Hours: The average sleep duration is roughly 6.5 hours. This ranges from a low of 3 hours to a high of 10 hours.
- Exercise Frequency: On average, students exercise about 3 times per week. Some don't exercise at all (0 times), while the most active exercise 6 times a week.
- Mental Health Rating: The average self-reported mental health rating is 5.4 out of 10. Ratings span from 1 (very poor) to 10 (excellent).
- Exam Score: The average exam score is 69.6 out of 100. Scores range widely from a low of 18.4 to a perfect 100.
'''
)

st.code(
'''
students_numeric = students.drop(['student_id','gender','part_time_job','diet_quality','parental_education_level','internet_quality','extracurricular_participation'], axis=1)
students_numeric.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(students_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()
'''
)

students_numeric = students.drop(['student_id','gender','part_time_job','diet_quality','parental_education_level','internet_quality','extracurricular_participation'], axis=1)
st.table(students_numeric.corr())
plt.figure(figsize=(8, 6))
sns.heatmap(students_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt)

st.markdown("From the heatmap of correlation, we can see correlation between exam score and study hours per day is 0,83. We can conclude that study hours per day has a strong positive correlation to exam score. More amout of hours students study per day strongly increasing exam score. Only study hours per day has a strong correlation to exam score. Study hours per day have a strong linearity to exam score. Correlation examscore between all of the variables except study hours per day didn't have a strong correlation (above 0,5), so, maybe if want to use linear regression, will be aadvised to use only study hours per day variables to preedict exam score. Next, for choosing categorical vairbales using anova.")

st.code(
'''
students_categoric = students[['gender','part_time_job','diet_quality','parental_education_level','internet_quality','extracurricular_participation','exam_score']]
model_anova = ols('exam_score ~ gender + part_time_job + diet_quality + parental_education_level + internet_quality + extracurricular_participation', data=students_categoric).fit()
aov_table = sm.stats.anova_lm(model_anova)
aov_table
'''
)

students_categoric = students[['gender','part_time_job','diet_quality','parental_education_level','internet_quality','extracurricular_participation','exam_score']]
model_anova = ols('exam_score ~ gender + part_time_job + diet_quality + parental_education_level + internet_quality + extracurricular_participation', data=students_categoric).fit()
aov_table = sm.stats.anova_lm(model_anova)
st.table(aov_table)

st.markdown("From Anova, there's no categorical value has a statisticly different between groups by seeing the p-value is above alpha 0,05. So, we can conclude that there's no categorical variables are fit to be inserted to dummy regression. But, to test dummy regression, will be modeling dummy regression with all of the variables just to be sure are the model dan predict exam score by seeing value of mean square error. Dummy regression is ways to predict numerical variables with numerical and categorical independent variables.")


st.code(
'''
# define X and y 
X = students.drop(['student_id','exam_score'], axis=1)
y = students['exam_score']

# splitting data into 80% training and 20% testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# dummy regression modeling
dummy_regr_mean = DummyRegressor(strategy="mean")
dummy_regr_mean.fit(X_train, y_train)
y_pred_mean = dummy_regr_mean.predict(X_test)
mse_mean = mean_squared_error(y_test, y_pred_mean)
mse_mean
'''
)

X = students.drop(['student_id','exam_score'], axis=1)
y = students['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dummy_regr_mean = DummyRegressor(strategy="mean")
dummy_regr_mean.fit(X_train, y_train)
y_pred_mean = dummy_regr_mean.predict(X_test)
mse_mean = mean_squared_error(y_test, y_pred_mean)
st.text(mse_mean)
st.markdown("The mse of dummy regression is 257,70. That mse is considered big value because the value of exam score between 18 and 100. Bigger the mse, the worse the model will become. A dummy regression model is not suitable for predicting exam scores. If the dummy regression are not suitable for predicting, try fit another regression, for example non-linear regression. Find the best non-linear distribution, the first thing to know is distributions of dependent variables. To find the best distribusion, i will use kolmogorov smirnov. The more lower KS statistic, increasing the chance of that distributions fit into the data. The distributions are normal, exponential, gamma, lognormal, weibull, uniform, t dist, chisquare and f distributions. ")




## fitted distribution
fitted_dist = '''
DISTRIBUTIONS = [
    sc.norm, sc.expon, sc.gamma, sc.lognorm, sc.weibull_min, sc.weibull_max, sc.uniform,
    sc.t, sc.chi2, sc.f # Common continuous distributions
]

best_distribution = None
best_params = None
best_ks_stat = np.inf # Kolmogorov-Smirnov statistic (lower is better)
best_p_value = 0

# Iterate over distributions, fit them, and perform K-S test
print("\nFitting distributions and performing K-S test:")
for distribution in DISTRIBUTIONS:
    try:
        with np.errstate(all='ignore'): # Suppress warnings during fitting
            params = distribution.fit(y)
        ks_stat, p_value = st.kstest(y, distribution.name, args=params)
        print(f"  {distribution.name:<15}: K-S Stat = {ks_stat:.4f}, P-value = {p_value:.4f}, Params = {params}")

        # Update best fit if this one is better (lower K-S statistic)
        # Also consider p-value (e.g., p_value > 0.05 as a threshold for a plausible fit)
        if ks_stat < best_ks_stat:
            best_distribution = distribution
            best_params = params
            best_ks_stat = ks_stat
            best_p_value = p_value

    except Exception as e:
        print(f"  Could not fit {distribution.name}: {e}")
        continue

if best_distribution:
    print(f"\nBest fitting distribution (lowest K-S statistic): {best_distribution.name}")
    print(f"  K-S Statistic: {best_ks_stat:.4f}")
    print(f"  P-value: {best_p_value:.4f} (If low, e.g. <0.05, the fit might not be good despite lowest K-S stat)")
    print(f"  Estimated Parameters: {best_params}")

    # Plot histogram of data and PDF of best-fitting distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins='auto', density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Data Histogram')

    # Generate x-axis values for plotting PDF
    xmin, xmax = plt.xlim()
    x_pdf = np.linspace(xmin, xmax, 100)
    pdf_fitted = best_distribution.pdf(x_pdf, *best_params)
    plt.plot(x_pdf, pdf_fitted, 'r-', lw=2, label=f'{best_distribution.name} PDF')

    plt.title(f'Data Histogram and Best Fit PDF ({best_distribution.name})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()
else:
    print("\nCould not find a suitable distribution or fitting failed for all candidates.")
'''
st.code(fitted_dist)
st.image("ks_par.png")
st.image("dist.png")

st.markdown("The best fitted probability density function of exam score are weibull min. So, predicting exam score can use weibull regression. However, weibull regression are meant to survival analysis, and this data was originally not for survival analysis. The hope for this data is to find the best way to predict exam score. So, i think the best modeling exam score are using linear regression with study hours per day. For visualisation dashboard, i will use google looker for descriptive and visualization in this [link](https://lookerstudio.google.com/reporting/9b5354c8-fff9-4645-83cc-99bb9eeb3ce4).")



st.subheader("Evaluation and Discussion")
st.markdown('''
From this analysis, that exam score can be predicted by study hours per day and mental helath rating from scatter plot (dashboard). The limitation of this data is that the data is not suitable for regular linear regression if want to use all of the data. Linear regression will be the best model if only use 2 variables, study hours per day and mental heallth rating. The other variables have a weak correlation. The data need to be converted to survival analysis structure to make the best fit model (y is weibull distribution), but the data is not originally for survival analysis. The dummy regression that was built are not that good when the MSE are about 250 from originally exam score have rage between 18-100. 
''')


