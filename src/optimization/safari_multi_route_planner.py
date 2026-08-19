import json
import pandas as pd
import numpy as np
import requests
import pulp
import folium
import os
from typing import Dict, List, Tuple, Optional
import time

class MultiRouteOSRMPlanner:
    """برنامه‌ریز مسیرهای چندگانه با OSRM"""
    
    def __init__(self):
        self.base_url = "http://router.project-osrm.org/route/v1/driving/"
        self.route_colors = ['green', 'red', 'blue', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
    
    def get_route(self, coordinates: List[Tuple[float, float]], color: str = 'blue') -> Optional[Dict]:
        """دریافت مسیر از OSRM برای چندین نقطه"""
        try:
            if len(coordinates) < 2:
                return None
            
            # ساخت رشته مختصات
            coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
            url = f"{self.base_url}{coords_str}"
            params = {
                'overview': 'full',
                'geometries': 'geojson'
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
                        'color': color,
                        'waypoints': data['waypoints']
                    }
            return None
        except Exception as e:
            print(f"OSRM route error: {e}")
            return None

class AdvancedSafariOptimizer:
    """بهینه‌ساز پیشرفته سافاری با چندین گزینه مسیر"""
    
    def __init__(self):
        self.lion_locations = self._load_comprehensive_lion_data()
        self.start_locations = {
            'Cairo, Egypt': (30.0444, 31.2357),
            'Nairobi, Kenya': (-1.286389, 36.817223),
            'Johannesburg, South Africa': (-26.2044, 28.0456),
            'Dar es Salaam, Tanzania': (-6.8235, 39.2695)
        }
    
    def _load_comprehensive_lion_data(self) -> Dict:
        """بارگذاری داده‌های جامع شیرها"""
        return {
            'Masai Mara, Kenya': {
                'coords': (-1.4066, 34.9066),
                'cost': 800, 'lion_density': 0.9, 'lion_count': 15,
                'description': 'Famous for big cat sightings and great migration',
                'best_season': 'July-October',
                'highlights': 'Big Five, Great Migration, Maasai Culture'
            },
            'Serengeti, Tanzania': {
                'coords': (-2.3333, 34.8333),
                'cost': 900, 'lion_density': 0.85, 'lion_count': 12,
                'description': 'Vast plains with high lion population', 
                'best_season': 'June-October',
                'highlights': 'River Crossings, Large Prides, Balloon Safaris'
            },
            'Kruger NP, South Africa': {
                'coords': (-24.0, 31.5),
                'cost': 700, 'lion_density': 0.8, 'lion_count': 10,
                'description': 'Excellent infrastructure and guaranteed sightings',
                'best_season': 'May-September',
                'highlights': 'Self-Drive Safaris, Luxury Lodges, Rhino Sightings'
            },
            'Okavango Delta, Botswana': {
                'coords': (-19.0, 23.0), 
                'cost': 1200, 'lion_density': 0.75, 'lion_count': 8,
                'description': 'Unique water-based lion sightings',
                'best_season': 'July-October',
                'highlights': 'Mokoro Trips, Swimming Lions, Bird Watching'
            },
            'Samburu, Kenya': {
                'coords': (0.5255, 37.5914),
                'cost': 600, 'lion_density': 0.7, 'lion_count': 6,
                'description': 'Less crowded with unique wildlife',
                'best_season': 'January-March & July-October',
                'highlights': 'Special Five, Desert Adaptation, Cultural Villages'
            },
            'Ngorongoro, Tanzania': {
                'coords': (-3.1874, 35.5762),
                'cost': 850, 'lion_density': 0.88, 'lion_count': 14,
                'description': 'Crater ecosystem with high density',
                'best_season': 'June-September', 
                'highlights': 'World Heritage Site, High Density, Stunning Views'
            }
        }
    
    def generate_route_options(self, start_location: str, budget: float) -> List[Dict]:
        """ایجاد چندین گزینه مسیر بر اساس بودجه"""
        print(f"\n🔄 Generating route options for ${budget:,} budget...")
        
        route_options = []
        
        # گزینه ۱: بهینه‌سازی برای حداکثر شیرها
        option1 = self._optimize_for_max_lions(start_location, budget)
        if option1:
            route_options.append({**option1, 'name': 'Maximum Lions', 'color': 'green'})
        
        # گزینه ۲: بهینه‌سازی برای تجربه متنوع
        option2 = self._optimize_for_diversity(start_location, budget)
        if option2:
            route_options.append({**option2, 'name': 'Diverse Experience', 'color': 'red'})
        
        # گزینه ۳: بهینه‌سازی برای کمترین هزینه
        option3 = self._optimize_for_budget(start_location, budget)
        if option3:
            route_options.append({**option3, 'name': 'Budget Friendly', 'color': 'blue'})
        
        # گزینه ۴: سافاری کلاسیک
        option4 = self._classic_safari_route(start_location, budget)
        if option4:
            route_options.append({**option4, 'name': 'Classic Safari', 'color': 'purple'})
        
        return route_options
    
    def _optimize_for_max_lions(self, start_location: str, budget: float) -> Optional[Dict]:
        """بهینه‌سازی برای بیشترین تعداد شیر"""
        model = pulp.LpProblem("Max_Lions_Optimization", pulp.LpMaximize)
        locations = list(self.lion_locations.keys())
        
        x = pulp.LpVariable.dicts("visit", locations, cat='Binary')
        objective = pulp.lpSum([x[loc] * self.lion_locations[loc]['lion_count'] for loc in locations])
        model += objective
        
        model += pulp.lpSum([x[loc] * self.lion_locations[loc]['cost'] for loc in locations]) <= budget
        model += pulp.lpSum([x[loc] for loc in locations]) <= 4
        
        model.solve()
        
        if pulp.LpStatus[model.status] == 'Optimal':
            selected = [loc for loc in locations if x[loc].value() == 1]
            return self._create_route_dict(selected, 'Maximum Lions')
        return None
    
    def _optimize_for_diversity(self, start_location: str, budget: float) -> Optional[Dict]:
        """بهینه‌سازی برای متنوع‌ترین تجربه"""
        # انتخاب موقعیت‌های مختلف از کشورهای مختلف
        countries = {}
        for loc, data in self.lion_locations.items():
            country = loc.split(', ')[1]
            if country not in countries:
                countries[country] = []
            countries[country].append((loc, data))
        
        selected = []
        remaining_budget = budget
        
        for country in countries:
            if remaining_budget <= 0:
                break
            # انتخاب بهترین موقعیت از هر کشور
            best_loc = max(countries[country], key=lambda x: x[1]['lion_density'])
            if best_loc[1]['cost'] <= remaining_budget:
                selected.append(best_loc[0])
                remaining_budget -= best_loc[1]['cost']
        
        if selected:
            return self._create_route_dict(selected, 'Diverse Experience')
        return None
    
    def _optimize_for_budget(self, start_location: str, budget: float) -> Optional[Dict]:
        """بهینه‌سازی برای کمترین هزینه"""
        # انتخاب موقعیت‌های با بهترین ارزش (شیر به ازای هزینه)
        locations = []
        for loc, data in self.lion_locations.items():
            value = data['lion_count'] / data['cost']
            locations.append((loc, data, value))
        
        # مرتب‌سازی بر اساس ارزش
        locations.sort(key=lambda x: x[2], reverse=True)
        
        selected = []
        remaining_budget = budget
        
        for loc, data, value in locations:
            if data['cost'] <= remaining_budget:
                selected.append(loc)
                remaining_budget -= data['cost']
        
        if selected:
            return self._create_route_dict(selected, 'Budget Friendly')
        return None
    
    def _classic_safari_route(self, start_location: str, budget: float) -> Optional[Dict]:
        """مسیر سافاری کلاسیک"""
        classic_routes = {
            'Cairo, Egypt': ['Masai Mara, Kenya', 'Serengeti, Tanzania'],
            'Nairobi, Kenya': ['Masai Mara, Kenya', 'Samburu, Kenya'],
            'Johannesburg, South Africa': ['Kruger NP, South Africa', 'Okavango Delta, Botswana'],
            'Dar es Salaam, Tanzania': ['Serengeti, Tanzania', 'Ngorongoro, Tanzania']
        }
        
        selected = classic_routes.get(start_location, ['Masai Mara, Kenya'])
        total_cost = sum([self.lion_locations[loc]['cost'] for loc in selected])
        
        if total_cost <= budget:
            return self._create_route_dict(selected, 'Classic Safari')
        return None
    
    def _create_route_dict(self, locations: List[str], route_type: str) -> Dict:
        """ایجاد دیکشنری مسیر"""
        total_cost = sum([self.lion_locations[loc]['cost'] for loc in locations])
        total_lions = sum([self.lion_locations[loc]['lion_count'] for loc in locations])
        
        itinerary = []
        for i, location in enumerate(locations, 1):
            loc_data = self.lion_locations[location]
            itinerary.append({
                'day': i,
                'location': location,
                'country': location.split(', ')[1],
                'coordinates': loc_data['coords'],
                'cost': loc_data['cost'],
                'estimated_lions': loc_data['lion_count'],
                'description': loc_data['description'],
                'best_season': loc_data['best_season'],
                'highlights': loc_data['highlights']
            })
        
        return {
            'route_type': route_type,
            'selected_locations': locations,
            'itinerary': itinerary,
            'total_cost': total_cost,
            'total_estimated_lions': total_lions,
            'days_required': len(locations)
        }

class ComprehensiveSafariMap:
    """نقشه جامع سافاری با تمام اطلاعات و مسیرهای چندگانه"""
    
    def __init__(self):
        self.route_planner = MultiRouteOSRMPlanner()
    
    def create_complete_safari_map(self, route_options: List[Dict], original_map_path: str = None) -> folium.Map:
        """ایجاد نقشه کامل با تمام اطلاعات"""
        print("🗺️ Creating comprehensive safari map with multiple routes...")
        
        # ایجاد نقشه جدید یا بارگذاری نقشه موجود
        if original_map_path and os.path.exists(original_map_path):
            # اگر نقشه اصلی موجود است، از آن استفاده کن
            safari_map = self._load_existing_map(original_map_path)
        else:
            # ایجاد نقشه جدید
            safari_map = folium.Map(
                location=[-8, 34],
                zoom_start=4,
                tiles='OpenStreetMap'
            )
        
        # اضافه کردن مسیرهای چندگانه
        self._add_multiple_routes(safari_map, route_options)
        
        # اضافه کردن راهنمای رنگ‌ها
        self._add_color_legend(safari_map, route_options)
        
        # اضافه کردن اطلاعات مسیرها
        self._add_routes_info_panel(safari_map, route_options)
        
        return safari_map
    
    def _load_existing_map(self, map_path: str) -> folium.Map:
        """بارگذاری نقشه موجود"""
        # برای سادگی، یک نقشه جدید ایجاد می‌کنیم
        # در عمل می‌توان از کتابخانه‌هایی برای خواندن HTML موجود استفاده کرد
        print("📂 Using base map configuration...")
        return folium.Map(
            location=[-8, 34],
            zoom_start=4,
            tiles='OpenStreetMap'
        )
    
    def _add_multiple_routes(self, map_obj: folium.Map, route_options: List[Dict]):
        """اضافه کردن مسیرهای چندگانه به نقشه"""
        for i, route in enumerate(route_options):
            color = route.get('color', self.route_planner.route_colors[i % len(self.route_planner.route_colors)])
            
            # جمع‌آوری مختصات برای این مسیر
            coordinates = []
            for stop in route['itinerary']:
                coords = stop['coordinates']
                coordinates.append((coords[0], coords[1]))
                
                # اضافه کردن marker برای هر توقف
                self._add_location_marker(map_obj, stop, color, route['route_type'])
            
            # دریافت مسیر از OSRM
            if len(coordinates) >= 2:
                route_data = self.route_planner.get_route(coordinates, color)
                if route_data:
                    self._add_route_to_map(map_obj, route_data, route)
    
    def _add_location_marker(self, map_obj: folium.Map, stop: Dict, color: str, route_type: str):
        """اضافه کردن marker موقعیت"""
        folium.Marker(
            stop['coordinates'],
            popup=f"""
            <div style="width: 300px;">
                <h4>🦁 {stop['location']}</h4>
                <b>Route:</b> {route_type}<br>
                <b>Day:</b> {stop['day']}<br>
                <b>Estimated Lions:</b> {stop['estimated_lions']}<br>
                <b>Cost:</b> ${stop['cost']:,}<br>
                <b>Best Season:</b> {stop['best_season']}<br>
                <b>Highlights:</b> {stop['highlights']}<br>
                <hr>
                <i>{stop['description']}</i>
            </div>
            """,
            tooltip=f"{route_type}: {stop['location']}",
            icon=folium.Icon(color=color, icon='paw', prefix='fa')
        ).add_to(map_obj)
    
    def _add_route_to_map(self, map_obj: folium.Map, route_data: Dict, route_info: Dict):
        """اضافه کردن مسیر به نقشه"""
        route_geometry = route_data['geometry']
        color = route_data['color']
        
        folium.PolyLine(
            locations=[(coord[1], coord[0]) for coord in route_geometry['coordinates']],
            color=color,
            weight=5,
            opacity=0.7,
            popup=f"""
            <div style="width: 250px;">
                <h4>🛣️ {route_info['route_type']}</h4>
                <b>Distance:</b> {route_data['distance_km']} km<br>
                <b>Duration:</b> {route_data['duration_hours']} hours<br>
                <b>Total Cost:</b> ${route_info['total_cost']:,}<br>
                <b>Estimated Lions:</b> {route_info['total_estimated_lions']}<br>
                <b>Destinations:</b> {len(route_info['itinerary'])}
            </div>
            """,
            tooltip=f"{route_info['route_type']} Route"
        ).add_to(map_obj)
    
    def _add_color_legend(self, map_obj: folium.Map, route_options: List[Dict]):
        """اضافه کردن راهنمای رنگ‌ها"""
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; background: white; 
        padding: 15px; border: 3px solid #2c3e50; border-radius: 10px; z-index: 9999; 
        font-size: 12px; max-width: 250px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 10px 0; color: #2c3e50;">🦁 Safari Route Colors</h4>
        '''
        
        for route in route_options:
            color = route.get('color', 'blue')
            legend_html += f'''
            <div style="margin: 5px 0;">
                <span style="background: {color}; width: 20px; height: 20px; 
                display: inline-block; border-radius: 3px; margin-right: 8px;"></span>
                <b>{route['route_type']}</b><br>
                <span style="margin-left: 28px; font-size: 11px; color: #666;">
                ${route['total_cost']:,} • {route['total_estimated_lions']} lions
                </span>
            </div>
            '''
        
        legend_html += '</div>'
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_routes_info_panel(self, map_obj: folium.Map, route_options: List[Dict]):
        """اضافه کردن پنل اطلاعات مسیرها"""
        info_html = '''
        <div style="position: fixed; top: 10px; left: 10px; background: white; 
        padding: 15px; border: 3px solid #27ae60; border-radius: 10px; z-index: 9999; 
        font-size: 12px; max-width: 350px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h3 style="margin: 0 0 10px 0; color: #2c3e50;">🦁 Multi-Route Safari Planner</h3>
        '''
        
        for i, route in enumerate(route_options, 1):
            color = route.get('color', 'blue')
            info_html += f'''
            <div style="margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid {color};">
                <b>{route['route_type']}</b><br>
                <span style="font-size: 11px;">
                💵 ${route['total_cost']:,} • 🦁 {route['total_estimated_lions']} lions • 📅 {route['days_required']} days<br>
                🗺️ {", ".join([stop['location'].split(",")[0] for stop in route['itinerary']])}
                </span>
            </div>
            '''
        
        info_html += '''
        <hr style="margin: 10px 0;">
        <div style="font-size: 11px; color: #666;">
        💡 <b>Click on routes</b> for details<br>
        💡 <b>Click on paw markers</b> for location info<br>
        💡 <b>Compare routes</b> using the color legend
        </div>
        </div>
        '''
        
        map_obj.get_root().html.add_child(folium.Element(info_html))

def main():
    """اجرای اصلی برنامه"""
    print("🦁 COMPREHENSIVE MULTI-ROUTE SAFARI PLANNER")
    print("=" * 55)
    
    # ایجاد نمونه‌ها
    safari_optimizer = AdvancedSafariOptimizer()
    map_creator = ComprehensiveSafariMap()
    
    try:
        # دریافت اطلاعات از کاربر
        print("\n🎯 Let's create multiple safari route options for you!")
        
        print("\n📍 Select your starting location:")
        start_options = list(safari_optimizer.start_locations.keys())
        for i, location in enumerate(start_options, 1):
            print(f"   {i}. {location}")
        
        start_choice = input("Enter choice (1-4): ").strip()
        start_location = start_options[int(start_choice)-1] if start_choice.isdigit() and 1 <= int(start_choice) <= 4 else start_options[0]
        
        print(f"\n💰 Enter your safari budget (USD):")
        print("   With $3,000 you can get multiple excellent route options!")
        budget = float(input("Your budget: $"))
        
        # تولید گزینه‌های مسیر
        route_options = safari_optimizer.generate_route_options(start_location, budget)
        
        # نمایش نتایج
        print("\n" + "=" * 60)
        print("🎉 YOUR SAFARI ROUTE OPTIONS")
        print("=" * 60)
        
        print(f"\n📍 Starting from: {start_location}")
        print(f"💰 Budget: ${budget:,}")
        print(f"🦁 Generated {len(route_options)} route options:")
        
        for i, route in enumerate(route_options, 1):
            print(f"\n{i}. {route['route_type']} ({route['color'].upper()})")
            print(f"   💵 Total cost: ${route['total_cost']:,}")
            print(f"   🦁 Estimated lions: {route['total_estimated_lions']}")
            print(f"   📅 Days required: {route['days_required']}")
            print(f"   🗺️ Route: {' → '.join([stop['location'].split(',')[0] for stop in route['itinerary']])}")
        
        # ایجاد نقشه جامع
        print(f"\n🗺️ Creating comprehensive map with all routes...")
        
        # استفاده از نقشه موجود اگر available باشد
        existing_map_path = "africa_lion_map_llm.html"
        
        safari_map = map_creator.create_complete_safari_map(
            route_options, 
            existing_map_path if os.path.exists(existing_map_path) else None
        )
        
        # ذخیره خروجی‌ها
        output_data = {
            'start_location': start_location,
            'budget': budget,
            'route_options': route_options,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'recommendations': {
                'best_value': max(route_options, key=lambda x: x['total_estimated_lions'] / x['total_cost']) if route_options else None,
                'most_diverse': max(route_options, key=lambda x: len(x['selected_locations'])) if route_options else None,
                'budget_friendly': min(route_options, key=lambda x: x['total_cost']) if route_options else None
            }
        }
        
        # ذخیره JSON
        with open('multi_route_safari_plan.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # ذخیره نقشه
        map_filename = 'comprehensive_multi_route_safari_map.html'
        safari_map.save(map_filename)
        
        print(f"\n✅ SUCCESS! Your multi-route safari plan is ready!")
        print(f"📁 Files created:")
        print(f"   • {map_filename} - Interactive map with ALL routes and information")
        print(f"   • multi_route_safari_plan.json - Complete route options data")
        
        print(f"\n🎯 What you'll see on the map:")
        print(f"   🟢 GREEN ROUTE - Maximum Lions (most lions for your budget)")
        print(f"   🔴 RED ROUTE - Diverse Experience (different countries/ecosystems)") 
        print(f"   🔵 BLUE ROUTE - Budget Friendly (best value)")
        print(f"   🟣 PURPLE ROUTE - Classic Safari (proven popular route)")
        
        print(f"\n💡 Map features:")
        print(f"   • All original lion sighting data preserved")
        print(f"   • Multiple colored routes with detailed information")
        print(f"   • Color legend and route info panel")
        print(f"   • Click any route or marker for details")
        
        print(f"\n🎊 With ${budget:,}, you have {len(route_options)} excellent options!")
        print(f"   Open '{map_filename}' to compare all routes visually!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()