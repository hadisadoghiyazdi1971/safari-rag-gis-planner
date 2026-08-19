import json
import pandas as pd
import numpy as np
import requests
import pulp
import folium
from folium.features import DivIcon
import os
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Tuple, Optional

class OSRMRoutePlanner:
    """برنامه‌ریز مسیر با استفاده از OSRM"""
    
    def __init__(self):
        self.base_url = "http://router.project-osrm.org/route/v1/driving/"
    
    def get_route(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> Optional[Dict]:
        """دریافت مسیر از OSRM"""
        try:
            # فرمت: lon,lat
            url = f"{self.base_url}{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'true'
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 'Ok':
                    route = data['routes'][0]
                    return {
                        'distance_km': round(route['distance'] / 1000, 2),
                        'duration_hours': round(route['duration'] / 3600, 2),
                        'geometry': route['geometry'],
                        'waypoints': data['waypoints']
                    }
            return None
        except Exception as e:
            print(f"OSRM API error: {e}")
            return None
    
    def get_route_between_multiple_points(self, coordinates: List[Tuple[float, float]]) -> Optional[Dict]:
        """دریافت مسیر بین چندین نقطه"""
        try:
            if len(coordinates) < 2:
                return None
            
            # ساخت رشته مختصات برای OSRM
            coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
            url = f"{self.base_url}{coords_str}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'true'
            }
            
            response = requests.get(url, params=params, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 'Ok':
                    route = data['routes'][0]
                    return {
                        'distance_km': round(route['distance'] / 1000, 2),
                        'duration_hours': round(route['duration'] / 3600, 2),
                        'geometry': route['geometry'],
                        'waypoints': data['waypoints'],
                        'legs': route['legs'] if 'legs' in route else []
                    }
            return None
        except Exception as e:
            print(f"OSRM multi-point error: {e}")
            return None

class SafariRouteOptimizer:
    """بهینه‌ساز مسیر سافاری با برنامه‌ریزی خطی"""
    
    def __init__(self):
        # موقعیت‌های محبوب برای مشاهده شیر
        self.lion_locations = {
            'Masai Mara, Kenya': {
                'coords': (-1.4066, 34.9066),
                'cost': 800,
                'lion_density': 0.9,
                'description': 'Famous for big cat sightings and great migration'
            },
            'Serengeti, Tanzania': {
                'coords': (-2.3333, 34.8333),
                'cost': 900,
                'lion_density': 0.85,
                'description': 'Vast plains with high lion population'
            },
            'Kruger NP, South Africa': {
                'coords': (-24.0, 31.5),
                'cost': 700,
                'lion_density': 0.8,
                'description': 'Excellent infrastructure and guaranteed sightings'
            },
            'Okavango Delta, Botswana': {
                'coords': (-19.0, 23.0),
                'cost': 1200,
                'lion_density': 0.75,
                'description': 'Unique water-based lion sightings'
            },
            'Samburu, Kenya': {
                'coords': (0.5255, 37.5914),
                'cost': 600,
                'lion_density': 0.7,
                'description': 'Less crowded with unique wildlife'
            },
            'Ngorongoro, Tanzania': {
                'coords': (-3.1874, 35.5762),
                'cost': 850,
                'lion_density': 0.88,
                'description': 'Crater ecosystem with high density'
            }
        }
        
        # موقعیت‌های شروع
        self.start_locations = {
            'Cairo, Egypt': (30.0444, 31.2357),
            'Nairobi, Kenya': (-1.286389, 36.817223),
            'Johannesburg, South Africa': (-26.2044, 28.0456),
            'Cape Town, South Africa': (-33.9249, 18.4241)
        }
    
    def optimize_route(self, start_loc: str, budget: float, max_destinations: int = 4) -> Dict:
        """بهینه‌سازی مسیر با محدودیت بودجه"""
        
        locations = list(self.lion_locations.keys())
        model = pulp.LpProblem("Safari_Route_Optimization", pulp.LpMaximize)
        
        # متغیرهای تصمیم
        x = pulp.LpVariable.dicts("visit", locations, cat='Binary')
        
        # تابع هدف: بیشینه کردن تراکم شیرها
        objective = pulp.lpSum([x[loc] * self.lion_locations[loc]['lion_density'] for loc in locations])
        model += objective
        
        # محدودیت بودجه
        model += pulp.lpSum([x[loc] * self.lion_locations[loc]['cost'] for loc in locations]) <= budget
        
        # محدودیت تعداد مقاصد
        model += pulp.lpSum([x[loc] for loc in locations]) <= max_destinations
        
        # حل مدل
        model.solve()
        
        # استخراج نتایج
        selected_locations = [loc for loc in locations if x[loc].value() == 1]
        
        return {
            'selected_locations': selected_locations,
            'total_locations': len(selected_locations),
            'total_lion_density': sum([self.lion_locations[loc]['lion_density'] for loc in selected_locations]),
            'total_cost': sum([self.lion_locations[loc]['cost'] for loc in selected_locations]),
            'status': pulp.LpStatus[model.status],
            'objective_value': pulp.value(model.objective)
        }

class InteractiveSafariPlanner:
    """برنامه‌ریز تعاملی سافاری با نمایش نقشه"""
    
    def __init__(self):
        self.route_planner = OSRMRoutePlanner()
        self.optimizer = SafariRouteOptimizer()
        
    def start_interactive_planning(self):
        """شروع برنامه‌ریزی تعاملی"""
        print("🦁 Welcome to Interactive Safari Route Planner!")
        print("=" * 50)
        
        # جمع‌آوری اطلاعات کاربر
        user_prefs = self._get_user_preferences()
        
        # بهینه‌سازی مسیر
        optimized_route = self.optimizer.optimize_route(
            user_prefs['start_location'],
            user_prefs['budget'],
            user_prefs['max_destinations']
        )
        
        # ایجاد نقشه تعاملی
        safari_map = self._create_interactive_map(user_prefs, optimized_route)
        
        # نمایش نتایج
        self._display_optimization_results(user_prefs, optimized_route)
        
        return safari_map, optimized_route
    
    def _get_user_preferences(self) -> Dict:
        """دریافت ترجیحات کاربر"""
        print("\nLet's plan your lion safari adventure! 🦁")
        
        # موقعیت شروع
        print("\n📍 Select your starting location:")
        for i, (name, coords) in enumerate(self.optimizer.start_locations.items(), 1):
            print(f"   {i}. {name}")
        
        start_choice = input("Enter choice (1-4): ").strip()
        start_location = self._get_choice_result(start_choice, list(self.optimizer.start_locations.keys()))
        
        # بودجه
        print(f"\n💰 What's your total safari budget? (USD)")
        budget = float(input("Enter budget: "))
        
        # تعداد مقاصد
        print(f"\n🎯 How many safari destinations do you want to visit?")
        max_destinations = int(input("Enter number (2-6): "))
        
        return {
            'start_location': start_location,
            'budget': budget,
            'max_destinations': max_destinations
        }
    
    def _get_choice_result(self, choice: str, options: List[str]) -> str:
        """تبدیل انتخاب کاربر به مقدار"""
        choice_map = {str(i+1): option for i, option in enumerate(options)}
        return choice_map.get(choice, options[0])
    
    def _create_interactive_map(self, user_prefs: Dict, optimized_route: Dict) -> folium.Map:
        """ایجاد نقشه تعاملی با مسیرهای OSRM"""
        print("\n🗺️ Creating interactive safari map...")
        
        # ایجاد نقشه پایه
        safari_map = folium.Map(
            location=[-8, 34],  # مرکز آفریقا
            zoom_start=4,
            tiles='OpenStreetMap'
        )
        
        start_coords = self.optimizer.start_locations[user_prefs['start_location']]
        selected_locations = optimized_route['selected_locations']
        
        # اضافه کردن marker موقعیت شروع
        folium.Marker(
            start_coords,
            popup=f"🚀 Start: {user_prefs['start_location']}",
            tooltip="Starting Point",
            icon=folium.Icon(color='green', icon='home', prefix='fa')
        ).add_to(safari_map)
        
        # جمع‌آوری مختصات برای مسیر
        all_coordinates = [start_coords]
        location_info = []
        
        for i, location in enumerate(selected_locations, 1):
            loc_data = self.optimizer.lion_locations[location]
            coords = loc_data['coords']
            all_coordinates.append(coords)
            
            # اضافه کردن marker مقصد
            folium.Marker(
                coords,
                popup=f"""
                🦁 {location}<br>
                Lion Density: {loc_data['lion_density']*100}%<br>
                Cost: ${loc_data['cost']}<br>
                {loc_data['description']}
                """,
                tooltip=f"Destination {i}: {location}",
                icon=folium.Icon(color='red', icon='paw', prefix='fa')
            ).add_to(safari_map)
            
            location_info.append({
                'name': location,
                'coords': coords,
                'lion_density': loc_data['lion_density'],
                'cost': loc_data['cost']
            })
        
        # دریافت مسیر از OSRM
        if len(all_coordinates) >= 2:
            route_data = self.route_planner.get_route_between_multiple_points(all_coordinates)
            
            if route_data:
                # اضافه کردن مسیر به نقشه
                route_geometry = route_data['geometry']
                
                folium.PolyLine(
                    locations=[(coord[1], coord[0]) for coord in route_geometry['coordinates']],
                    color='blue',
                    weight=4,
                    opacity=0.7,
                    popup=f"""
                    🛣️ Safari Route<br>
                    Total Distance: {route_data['distance_km']} km<br>
                    Estimated Duration: {route_data['duration_hours']} hours<br>
                    Stops: {len(selected_locations)} destinations
                    """,
                    tooltip="Optimized Safari Route"
                ).add_to(safari_map)
                
                # اضافه کردن اطلاعات مسیر
                route_info = f"""
                <div style="position: fixed; top: 10px; left: 50px; background: white; 
                padding: 10px; border: 2px solid blue; z-index: 9999; font-size: 12px;">
                <h4>🦁 Safari Route Info</h4>
                <b>Start:</b> {user_prefs['start_location']}<br>
                <b>Total Distance:</b> {route_data['distance_km']} km<br>
                <b>Duration:</b> {route_data['duration_hours']} hours<br>
                <b>Destinations:</b> {len(selected_locations)}<br>
                <b>Total Budget:</b> ${user_prefs['budget']:,}<br>
                <b>Route Cost:</b> ${optimized_route['total_cost']:,}
                </div>
                """
                safari_map.get_root().html.add_child(folium.Element(route_info))
        
        # اضافه کردن راهنمای نقشه
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; background: white; 
        padding: 10px; border: 2px solid grey; z-index: 9999; font-size: 12px;">
        <h4>Map Legend</h4>
        <p>🟢 <b>Start Point</b> - Your starting location</p>
        <p>🔴 <b>Lion Destinations</b> - Safari locations</p>
        <p>🔵 <b>Blue Line</b> - Optimized route from OSRM</p>
        <p>📊 <b>Top Left</b> - Route information</p>
        </div>
        '''
        safari_map.get_root().html.add_child(folium.Element(legend_html))
        
        # اضافه کردن کنترل لایه‌ها
        folium.LayerControl().add_to(safari_map)
        
        return safari_map
    
    def _display_optimization_results(self, user_prefs: Dict, optimized_route: Dict):
        """نمایش نتایج بهینه‌سازی"""
        print("\n" + "=" * 60)
        print("🎯 OPTIMIZED SAFARI ROUTE RESULTS")
        print("=" * 60)
        
        print(f"\n📍 Starting from: {user_prefs['start_location']}")
        print(f"💰 Budget: ${user_prefs['budget']:,}")
        print(f"🎯 Max destinations: {user_prefs['max_destinations']}")
        
        print(f"\n🦁 Selected Safari Destinations:")
        for i, location in enumerate(optimized_route['selected_locations'], 1):
            loc_data = self.optimizer.lion_locations[location]
            print(f"   {i}. {location}")
            print(f"      Lion Density: {loc_data['lion_density']*100}%")
            print(f"      Cost: ${loc_data['cost']:,}")
            print(f"      Description: {loc_data['description']}")
        
        print(f"\n📊 Optimization Summary:")
        print(f"   • Total destinations: {optimized_route['total_locations']}")
        print(f"   • Total lion density score: {optimized_route['total_lion_density']:.2f}")
        print(f"   • Total route cost: ${optimized_route['total_cost']:,}")
        print(f"   • Budget remaining: ${user_prefs['budget'] - optimized_route['total_cost']:,}")
        print(f"   • Optimization status: {optimized_route['status']}")
        
        # تحلیل بودجه
        budget_usage = (optimized_route['total_cost'] / user_prefs['budget']) * 100
        if budget_usage <= 80:
            print(f"   ✅ Budget usage: {budget_usage:.1f}% - Excellent!")
        elif budget_usage <= 100:
            print(f"   ⚠️ Budget usage: {budget_usage:.1f}% - Within budget")
        else:
            print(f"   ❌ Budget usage: {budget_usage:.1f}% - Over budget")

def generate_sample_data():
    """ایجاد داده‌های نمونه برای تست"""
    sample_lion_sightings = [
        {
            'filename': 'lion_maasai_mara.jpg',
            'location': 'Masai Mara, Kenya', 
            'country': 'Kenya',
            'latitude': -1.4066,
            'longitude': 34.9066,
            'blip_description': 'A pride of lions resting in the savanna',
            'llm_description': 'A magnificent pride of African lions basking in the golden savanna sunlight',
            'score': 10
        },
        {
            'filename': 'lion_serengeti.jpg',
            'location': 'Serengeti, Tanzania',
            'country': 'Tanzania', 
            'latitude': -2.3333,
            'longitude': 34.8333,
            'blip_description': 'Male lion on rocky outcrop',
            'llm_description': 'A dominant male lion surveying his territory from a rocky kopje',
            'score': 8
        }
    ]
    
    # ذخیره داده‌های نمونه
    df = pd.DataFrame(sample_lion_sightings)
    df.to_csv('sample_lion_data.csv', index=False)
    print("✅ Sample data created: sample_lion_data.csv")

def main():
    """اجرای اصلی برنامه"""
    
    # ایجاد داده‌های نمونه اگر وجود ندارند
    if not os.path.exists('sample_lion_data.csv'):
        generate_sample_data()
    
    # ایجاد برنامه‌ریز
    planner = InteractiveSafariPlanner()
    
    try:
        # شروع برنامه‌ریزی تعاملی
        safari_map, optimized_route = planner.start_interactive_planning()
        
        # ذخیره نقشه
        map_filename = "interactive_safari_route_map.html"
        safari_map.save(map_filename)
        
        print(f"\n🗺️ Interactive map saved: {map_filename}")
        print("   - Open this file in your browser to see the optimized route")
        print("   - Blue line shows the actual OSRM calculated route")
        print("   - Click on markers for detailed information")
        print("   - Top-left panel shows route statistics")
        
        # ذخیره نتایج بهینه‌سازی
        with open('safari_optimization_results.json', 'w', encoding='utf-8') as f:
            json.dump(optimized_route, f, indent=2, ensure_ascii=False)
        print(f"💾 Optimization results saved: safari_optimization_results.json")
        
        print(f"\n🎉 Your lion safari is planned! Open '{map_filename}' to explore your route.")
        
    except KeyboardInterrupt:
        print("\n\n👋 Safari planning cancelled. Hope to see you on another adventure!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your internet connection for OSRM routing")

if __name__ == "__main__":
    main()