import json
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional
import pulp
from math import radians, sin, cos, sqrt, atan2
import folium
from datetime import datetime, timedelta

class SafariCostCalculator:
    """ماشین حساب هزینه‌های سفر Safari"""
    
    def __init__(self):
        self.cost_parameters = {
            'transportation': {
                'car_rental_per_km': 0.15,  # USD per km
                'fuel_per_liter': 1.2,      # USD
                'fuel_consumption': 8,      # km per liter
                'driver_guide_per_day': 80, # USD
            },
            'accommodation': {
                'budget_lodge': 50,         # USD per night
                'mid_range_lodge': 120,     # USD per night  
                'luxury_lodge': 300,        # USD per night
            },
            'activities': {
                'park_entry_per_day': 70,   # USD
                'game_drive_per_session': 40, # USD
                'guided_walk': 25,          # USD
            },
            'meals': {
                'budget_per_day': 25,       # USD
                'mid_range_per_day': 50,    # USD
                'luxury_per_day': 100,      # USD
            }
        }
    
    def calculate_trip_cost(self, distance_km: float, days: int, comfort_level: str = 'mid_range') -> Dict:
        """محاسبه هزینه کل سفر"""
        
        # انتخاب سطح راحتی
        accommodation_key = f"{comfort_level}_lodge"
        meals_key = f"{comfort_level}_per_day"
        
        # محاسبه هزینه‌ها
        transport_cost = distance_km * self.cost_parameters['transportation']['car_rental_per_km']
        fuel_cost = (distance_km / self.cost_parameters['transportation']['fuel_consumption']) * self.cost_parameters['transportation']['fuel_per_liter']
        driver_cost = days * self.cost_parameters['transportation']['driver_guide_per_day']
        
        accommodation_cost = days * self.cost_parameters['accommodation'][accommodation_key]
        meals_cost = days * self.cost_parameters['meals'][meals_key]
        activities_cost = (days * self.cost_parameters['activities']['park_entry_per_day'] + 
                          days * 2 * self.cost_parameters['activities']['game_drive_per_session'])
        
        total_cost = (transport_cost + fuel_cost + driver_cost + 
                     accommodation_cost + meals_cost + activities_cost)
        
        return {
            'total_cost_usd': round(total_cost, 2),
            'transportation': round(transport_cost + fuel_cost + driver_cost, 2),
            'accommodation': round(accommodation_cost, 2),
            'meals': round(meals_cost, 2),
            'activities': round(activities_cost, 2),
            'cost_per_day': round(total_cost / days, 2),
            'comfort_level': comfort_level
        }

class LionSightingOptimizer:
    """بهینه‌ساز مسیر برای مشاهده شیرها با استفاده از برنامه‌ریزی خطی"""
    
    def __init__(self, lion_data_file: str = "africa_descriptions_export.csv"):
        self.lion_data = pd.read_csv(lion_data_file)
        self.lion_locations = self._extract_lion_locations()
    
    def _extract_lion_locations(self) -> List[Dict]:
        """استخراج موقعیت‌های شیرها از داده‌ها"""
        lion_sightings = []
        
        for _, row in self.lion_data.iterrows():
            # جستجوی شیر در توضیحات
            description = str(row.get('llm_description', '') or row.get('blip_description', '')).lower()
            if 'lion' in description:
                try:
                    # استخراج مختصات از فایل اصلی (نیاز به تطبیق با داده‌های اصلی)
                    lion_sightings.append({
                        'location': row.get('location', 'Unknown'),
                        'country': row.get('country', 'Unknown'),
                        'description': description[:100] + '...' if len(description) > 100 else description,
                        'score': description.count('lion')  # امتیاز بر اساس تعداد تکرار
                    })
                except:
                    continue
        
        return sorted(lion_sightings, key=lambda x: x['score'], reverse=True)
    
    def optimize_route(self, start_location: str, budget: float, days: int, max_locations: int = 5):
        """بهینه‌سازی مسیر با محدودیت بودجه و زمان"""
        
        # داده‌های نمونه برای موقعیت‌ها (در عمل از API جغرافیایی استفاده کنید)
        location_data = {
            'Cairo': {'lat': 30.0444, 'lon': 31.2357, 'cost_multiplier': 1.0},
            'Masai Mara': {'lat': -1.4066, 'lon': 34.9066, 'cost_multiplier': 1.2},
            'Serengeti': {'lat': -2.3333, 'lon': 34.8333, 'cost_multiplier': 1.3},
            'Kruger': {'lat': -24.0, 'lon': 31.5, 'cost_multiplier': 1.1},
            'Okavango': {'lat': -19.0, 'lon': 23.0, 'cost_multiplier': 1.4}
        }
        
        # ایجاد مدل بهینه‌سازی
        model = pulp.LpProblem("Lion_Sighting_Optimization", pulp.LpMaximize)
        
        # متغیرهای تصمیم
        locations = list(location_data.keys())
        x = pulp.LpVariable.dicts("visit", locations, cat='Binary')
        
        # تابع هدف: بیشینه کردن تعداد موقعیت‌های شیر
        objective = pulp.lpSum([x[loc] for loc in locations])
        model += objective
        
        # محدودیت بودجه
        cost_per_location = 500  # USD تخمینی برای هر موقعیت
        model += pulp.lpSum([x[loc] * cost_per_location * location_data[loc]['cost_multiplier'] 
                           for loc in locations]) <= budget
        
        # محدودیت زمان
        model += pulp.lpSum([x[loc] for loc in locations]) <= min(max_locations, days)
        
        # حل مدل
        model.solve()
        
        # استخراج نتایج
        selected_locations = [loc for loc in locations if x[loc].value() == 1]
        
        return {
            'selected_locations': selected_locations,
            'total_locations': len(selected_locations),
            'status': pulp.LpStatus[model.status],
            'objective_value': pulp.value(model.objective)
        }

class OSRMRoutePlanner:
    """برنامه‌ریز مسیر با استفاده از OSRM"""
    
    def __init__(self):
        self.base_url = "http://router.project-osrm.org/route/v1/driving/"
    
    def get_route(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> Optional[Dict]:
        """دریافت مسیر از OSRM"""
        try:
            url = f"{self.base_url}{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 'Ok':
                    route = data['routes'][0]
                    return {
                        'distance_km': round(route['distance'] / 1000, 2),
                        'duration_hours': round(route['duration'] / 3600, 2),
                        'geometry': route['geometry']
                    }
            return None
        except Exception as e:
            print(f"OSRM API error: {e}")
            return None

class SafariTripPlanner:
    """سیستم اصلی برنامه‌ریزی سفر Safari"""
    
    def __init__(self):
        self.cost_calculator = SafariCostCalculator()
        self.lion_optimizer = LionSightingOptimizer()
        self.route_planner = OSRMRoutePlanner()
        
        # داده‌های موقعیت‌های محبوب شیر
        self.popular_lion_locations = {
            'Masai Mara, Kenya': {'coords': (-1.4066, 34.9066), 'lion_density': 'Very High'},
            'Serengeti, Tanzania': {'coords': (-2.3333, 34.8333), 'lion_density': 'Very High'},
            'Kruger National Park, South Africa': {'coords': (-24.0, 31.5), 'lion_density': 'High'},
            'Okavango Delta, Botswana': {'coords': (-19.0, 23.0), 'lion_density': 'High'},
            'Samburu, Kenya': {'coords': (0.5255, 37.5914), 'lion_density': 'Medium'},
            'Ngorongoro, Tanzania': {'coords': (-3.1874, 35.5762), 'lion_density': 'High'}
        }
    
    def start_conversation(self):
        """شروع مکالمه هوشمند با کاربر"""
        print("🦁 Welcome to African Lion Safari Planner!")
        print("=" * 50)
        
        # جمع‌آوری اطلاعات از کاربر
        user_data = self._collect_user_preferences()
        
        # برنامه‌ریزی سفر
        trip_plan = self._create_trip_plan(user_data)
        
        # نمایش نتایج
        self._display_trip_plan(trip_plan)
        
        return trip_plan
    
    def _collect_user_preferences(self) -> Dict:
        """جمع‌آوری ترجیحات کاربر"""
        print("\nLet's plan your perfect lion safari! 🦁")
        
        # موقعیت فعلی
        print("\n📍 Where are you starting from?")
        print("1. Cairo, Egypt (Default)")
        print("2. Nairobi, Kenya") 
        print("3. Johannesburg, South Africa")
        print("4. Other (please specify)")
        
        start_choice = input("Enter choice (1-4): ").strip()
        start_location = self._get_start_location(start_choice)
        
        # بودجه
        print(f"\n💰 What's your total budget for this safari? (USD)")
        print("We're currently in Egypt - do you know the current costs?")
        budget = float(input("Enter your budget in USD: "))
        
        # مدت سفر
        print(f"\n📅 How many days do you have for your safari?")
        days = int(input("Enter number of days: "))
        
        # سطح راحتی
        print(f"\n🏨 What comfort level do you prefer?")
        print("1. Budget (Camping, Basic Lodges)")
        print("2. Mid-range (Comfortable Lodges)") 
        print("3. Luxury (Premium Safari Lodges)")
        
        comfort_choice = input("Enter choice (1-3): ").strip()
        comfort_level = self._get_comfort_level(comfort_choice)
        
        return {
            'start_location': start_location,
            'budget': budget,
            'days': days,
            'comfort_level': comfort_level,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_start_location(self, choice: str) -> str:
        """تعیین موقعیت شروع"""
        locations = {
            '1': 'Cairo, Egypt',
            '2': 'Nairobi, Kenya',
            '3': 'Johannesburg, South Africa',
            '4': input("Please specify your starting location: ")
        }
        return locations.get(choice, 'Cairo, Egypt')
    
    def _get_comfort_level(self, choice: str) -> str:
        """تعیین سطح راحتی"""
        levels = {'1': 'budget', '2': 'mid_range', '3': 'luxury'}
        return levels.get(choice, 'mid_range')
    
    def _create_trip_plan(self, user_data: Dict) -> Dict:
        """ایجاد برنامه سفر"""
        print("\n🔄 Creating your optimized safari plan...")
        
        # بهینه‌سازی موقعیت‌ها
        optimization_result = self.lion_optimizer.optimize_route(
            user_data['start_location'],
            user_data['budget'],
            user_data['days']
        )
        
        # انتخاب بهترین موقعیت‌ها بر اساس بهینه‌سازی
        selected_locations = optimization_result['selected_locations']
        if not selected_locations:
            # اگر بهینه‌سازی نتیجه نداد، از موقعیت‌های پیش‌فرض استفاده کن
            selected_locations = ['Masai Mara, Kenya', 'Serengeti, Tanzania'][:min(2, user_data['days'])]
        
        # محاسبه مسافت و هزینه
        total_distance = 0
        route_geometry = []
        
        # محاسبه مسیر (ساده‌سازی شده)
        for i in range(len(selected_locations) - 1):
            start_loc = selected_locations[i]
            end_loc = selected_locations[i + 1]
            
            if start_loc in self.popular_lion_locations and end_loc in self.popular_lion_locations:
                start_coords = self.popular_lion_locations[start_loc]['coords']
                end_coords = self.popular_lion_locations[end_loc]['coords']
                
                route = self.route_planner.get_route(start_coords, end_coords)
                if route:
                    total_distance += route['distance_km']
                    route_geometry.append(route['geometry'])
        
        # محاسبه هزینه
        cost_breakdown = self.cost_calculator.calculate_trip_cost(
            total_distance, user_data['days'], user_data['comfort_level']
        )
        
        # ایجاد برنامه روزانه
        daily_itinerary = self._create_daily_itinerary(selected_locations, user_data['days'])
        
        return {
            'user_preferences': user_data,
            'optimization_result': optimization_result,
            'selected_locations': selected_locations,
            'total_distance_km': total_distance,
            'cost_breakdown': cost_breakdown,
            'daily_itinerary': daily_itinerary,
            'route_geometry': route_geometry,
            'lion_sighting_probability': self._calculate_sighting_probability(selected_locations)
        }
    
    def _create_daily_itinerary(self, locations: List[str], days: int) -> List[Dict]:
        """ایجاد برنامه روزانه"""
        itinerary = []
        
        for day in range(days):
            location_idx = min(day, len(locations) - 1)
            current_location = locations[location_idx]
            
            itinerary.append({
                'day': day + 1,
                'location': current_location,
                'activities': [
                    'Morning Game Drive (6:00-9:00)',
                    'Breakfast at Lodge',
                    'Rest & Wildlife Observation',
                    'Evening Game Drive (16:00-19:00)',
                    'Dinner & Safari Stories'
                ],
                'lion_spotting_tips': self._get_lion_spotting_tips(current_location)
            })
        
        return itinerary
    
    def _get_lion_spotting_tips(self, location: str) -> str:
        """نکات مشاهده شیر بر اساس موقعیت"""
        tips = {
            'Masai Mara, Kenya': "Look near riverbanks and kopjes in early morning",
            'Serengeti, Tanzania': "Check rocky outcrops and areas with wildebeest herds",
            'Kruger National Park, South Africa': "Search near waterholes during dry season",
            'Okavango Delta, Botswana': "Look for lions on islands and floodplains"
        }
        return tips.get(location, "Ask local guides for recent lion sightings")
    
    def _calculate_sighting_probability(self, locations: List[str]) -> float:
        """محاسبه احتمال مشاهده شیر"""
        total_prob = 0
        for location in locations:
            if location in self.popular_lion_locations:
                density = self.popular_lion_locations[location]['lion_density']
                prob_map = {'Very High': 0.9, 'High': 0.7, 'Medium': 0.5, 'Low': 0.3}
                total_prob += prob_map.get(density, 0.5)
        
        return round(total_prob / len(locations) * 100, 1) if locations else 0
    
    def _display_trip_plan(self, trip_plan: Dict):
        """نمایش برنامه سفر"""
        print("\n" + "=" * 60)
        print("🎯 YOUR OPTIMIZED LION SAFARI PLAN")
        print("=" * 60)
        
        user_prefs = trip_plan['user_preferences']
        cost = trip_plan['cost_breakdown']
        
        print(f"\n📍 Starting from: {user_prefs['start_location']}")
        print(f"💰 Budget: ${user_prefs['budget']:,}")
        print(f"📅 Duration: {user_prefs['days']} days")
        print(f"🏨 Comfort: {user_prefs['comfort_level'].replace('_', ' ').title()}")
        
        print(f"\n🦁 Selected Lion Locations:")
        for i, location in enumerate(trip_plan['selected_locations'], 1):
            print(f"   {i}. {location}")
        
        print(f"\n📊 Trip Statistics:")
        print(f"   • Total distance: {trip_plan['total_distance_km']} km")
        print(f"   • Lion sighting probability: {trip_plan['lion_sighting_probability']}%")
        print(f"   • Optimization status: {trip_plan['optimization_result']['status']}")
        
        print(f"\n💵 Cost Breakdown:")
        print(f"   • Transportation: ${cost['transportation']:,}")
        print(f"   • Accommodation: ${cost['accommodation']:,}")
        print(f"   • Meals: ${cost['meals']:,}")
        print(f"   • Activities: ${cost['activities']:,}")
        print(f"   • Total: ${cost['total_cost_usd']:,}")
        print(f"   • Cost per day: ${cost['cost_per_day']:,}")
        
        print(f"\n📅 Daily Itinerary:")
        for day_plan in trip_plan['daily_itinerary']:
            print(f"\n   Day {day_plan['day']}: {day_plan['location']}")
            print(f"   Tips: {day_plan['lion_spotting_tips']}")
        
        print(f"\n💡 Budget Analysis:")
        budget_ratio = cost['total_cost_usd'] / user_prefs['budget']
        if budget_ratio <= 0.8:
            print("   ✅ Great! Your budget is sufficient with good margin.")
        elif budget_ratio <= 1.0:
            print("   ⚠️ Your budget is tight but workable.")
        else:
            print("   ❌ Your budget may not be sufficient. Consider:")
            print("      - Reducing trip duration")
            print("      - Choosing budget accommodation")
            print("      - Focusing on fewer locations")
        
        print(f"\n🎉 Ready for your adventure! Contact safari operators in {user_prefs['start_location']} to finalize.")

def main():
    """اجرای اصلی برنامه"""
    planner = SafariTripPlanner()
    
    try:
        trip_plan = planner.start_conversation()
        
        # ذخیره برنامه
        with open('safari_trip_plan.json', 'w', encoding='utf-8') as f:
            json.dump(trip_plan, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Trip plan saved to 'safari_trip_plan.json'")
        
    except KeyboardInterrupt:
        print("\n\n👋 Safari planning cancelled. Hope to see you on another adventure!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()