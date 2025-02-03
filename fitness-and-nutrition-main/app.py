import random
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load datasets
exercise_df = pd.read_csv('new_cleaned_exercise_dataset.csv')
nutrition_df = pd.read_csv('neutritionData1.csv')

# Train a regression model to predict calorie intake
X_nutrition = nutrition_df[['Protein (g)', 'Carbs (g)', 'Fat (g)']]
y_nutrition = nutrition_df['Total Calories']
X_train, X_test, y_train, y_test = train_test_split(X_nutrition, y_nutrition, test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy = 100 - (mape * 100)

# Activity level mapping
activity_level_mapping = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very active": 1.9
}

def generate_workout_plan(targeted_muscle, difficulty, num_weeks):
    workout_plan = []
    for week in range(1, num_weeks + 1):
        for day in range(1, 8):
            filtered_exercises = exercise_df[
            (exercise_df['Targeted Muscle'].str.lower().str.strip() == targeted_muscle) & 
            (exercise_df['Difficulty Level'] <= difficulty)
            ]
            if filtered_exercises.empty:
                daily_exercises = ["No available exercises for this selection."]
            else:
                daily_exercises = random.sample(list(filtered_exercises['Exercise Name']), min(3, len(filtered_exercises)))
            workout_plan.append({
                'week': week,
                'day': day,
                'workout': [f"{exercise} - 3 sets x 8 reps" for exercise in daily_exercises]
            })
    return workout_plan

def generate_meal_plan(diet_type, daily_calories, num_weeks):
    meal_plan = []
    for week in range(1, num_weeks + 1):
        for day in range(1, 8):
            filtered_meals = nutrition_df[nutrition_df['Veg / Non-veg'].str.lower() == diet_type]
            if filtered_meals.empty:
                daily_meals = ["No available meals for this selection."]
            else:
                daily_meals = filtered_meals.sample(min(4, len(filtered_meals)))
            meal_plan.append({
                'week': week,
                'day': day,
                'diet': [
                    f"Breakfast: {daily_meals.iloc[0]['Recipe Name']} - {daily_meals.iloc[0]['Total Calories']} kcal",
                    f"Lunch: {daily_meals.iloc[1]['Recipe Name']} - {daily_meals.iloc[1]['Total Calories']} kcal",
                    f"Dinner: {daily_meals.iloc[2]['Recipe Name']} - {daily_meals.iloc[2]['Total Calories']} kcal",
                    f"Snack: {daily_meals.iloc[3]['Recipe Name']} - {daily_meals.iloc[3]['Total Calories']} kcal"
                ]
            })
    return meal_plan

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    name = request.form['name']
    height_cm = float(request.form['height'])
    weight_kg = float(request.form['weight'])
    goal = request.form['goal'].lower()
    activity_level_str = request.form['activity_level'].lower()
    targeted_muscle = request.form['targeted_muscle'].lower()
    difficulty = int(request.form['difficulty'])
    num_weeks = int(request.form['num_weeks'])
    diet_type = request.form['diet_type'].lower()
    age = int(request.form['age'])  # Get age from the form
    gender = request.form['gender'].lower()  # Get gender from the form

    # Generate workout plan
    workout_plan = generate_workout_plan(targeted_muscle, difficulty, num_weeks)
    
    # Activity level mapping
    activity_level = activity_level_mapping.get(activity_level_str, 1.2)
    
    # Calculate BMR based on age and gender
    if gender == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5  # Mifflin-St Jeor formula for males
    elif gender == 'female':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161  # Mifflin-St Jeor formula for females
    
    tdee = bmr * activity_level

    # Calorie intake adjustments based on goal
    if goal == "bulking":
        daily_calories = tdee * 1.2
    elif goal == "cutting":
        daily_calories = tdee * 0.8
    else:
        daily_calories = tdee

    # Macronutrient breakdown
    protein = round((daily_calories * 0.3) / 4, 1)
    carbs = round((daily_calories * 0.4) / 4, 1)
    fats = round((daily_calories * 0.3) / 9, 1)

    # Generate meal plan
    meal_plan = generate_meal_plan(diet_type, daily_calories, num_weeks)

    # Merge workout plan and meal plan
    full_plan = []
    for i in range(len(workout_plan)):
        full_plan.append({
            'week': workout_plan[i]['week'],
            'day': workout_plan[i]['day'],
            'workout': workout_plan[i]['workout'],
            'diet': meal_plan[i]['diet']
        })

    # Render the result page
    return render_template('result.html', name=name, daily_calories=round(daily_calories, 2), 
                           protein=protein, carbs=carbs, fats=fats, plan=full_plan)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
