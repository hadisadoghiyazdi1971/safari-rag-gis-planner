import json
import pandas as pd
import numpy as np
import requests
import pulp
import folium
from folium.features import DivIcon
import os
from typing import Dict, List, Tuple, Optional
import time

class LLMStoryGenerator:
    """تولیدکننده داستان سفر با استفاده از LLM"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def generate_safari_story(self, itinerary: List[Dict], budget: float, total_lions_seen: int) -> Optional[str]:
        """تولید داستان سفر سافاری"""
        if not self.api_key:
            return self._generate_fallback_story(itinerary, budget, total_lions_seen)
        
        try:
            locations = [f"{stop['location']} ({stop['country']})" for stop in itinerary]
            locations_str = " → ".join(locations)
            
            prompt = f"""
You are a professional safari guide and storyteller. Create an engaging, vivid story about this African lion safari adventure:

**Safari Itinerary**: {locations_str}
**Total Budget**: ${budget:,}
**Estimated Lions Seen**: {total_lions_seen}
**Duration**: {len(itinerary)} days

Please write a captivating story with:
1. **Introduction** - The excitement of starting this adventure
2. **Daily Experiences** - What happens at each location, wildlife encounters
3. **Lion Encounters** - Specific lion sightings and behaviors observed
4. **Cultural Experiences** - Local culture and people met
5. **Conclusion** - Reflections on the journey and conservation message

Write in English, be descriptive and engaging, around 500-600 words. Use vivid imagery and make the reader feel like they're on the safari.
"""
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.8
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return self._generate_fallback_story(itinerary, budget, total_lions_seen)
                
        except Exception as e:
            print(f"LLM Story Error: {e}")
            return self._generate_fallback_story(itinerary, budget, total_lions_seen)
    
    def _generate_fallback_story(self, itinerary: List[Dict], budget: float, total_lions: int) -> str:
        """ایجاد داستان جایگزین در صورت عدم دسترسی به LLM"""
        locations = [stop['location'] for stop in itinerary]
        
        story = f"""The African Lion Safari Adventure

With a budget of ${budget:,}, our journey begins in search of the magnificent African lion across the most renowned wildlife destinations.

**Day 1: Arrival in {locations[0] if locations else 'Africa'}**
The African sun welcomes us as we begin our quest. The air is filled with anticipation and the distant calls of wildlife. Our experienced guides prepare us for the adventures ahead, sharing stories of lion prides and their territories.

"""
        
        for i, stop in enumerate(itinerary, 1):
            story += f"""
**Day {i}: Exploring {stop['location']}**
{stop.get('story_segment', 'We venture into the wild landscape, scanning the horizon for movement. The golden grass sways in the breeze as we search for the kings of the savanna.')}

"""
        
        story += f"""
**The Grand Finale**
Over {len(itinerary)} incredible days, we've witnessed approximately {total_lions} lions in their natural habitats. From majestic males surveying their domains to playful cubs learning to hunt, each encounter has been unforgettable.

This journey has been more than a safari - it's been a connection with nature's most magnificent predators and a reminder of the importance of conservation efforts across Africa.

Total lions encountered: {total_lions}
Destinations visited: {len(locations)}
Memories made: Countless
"""
        
        return story

class EnhancedSafariPlanner:
    """برنامه‌ریز پیشرفته سافاری با داستان‌سرایی"""
    
    def __init__(self):
        self.lion_locations = self._load_lion_data()
        self.start_locations = {
            'Cairo, Egypt': (30.0444, 31.2357),
            'Nairobi, Kenya': (-1.286389, 36.817223),
            'Johannesburg, South Africa': (-26.2044, 28.0456),
            'Dar es Salaam, Tanzania': (-6.8235, 39.2695)
        }
        self.story_generator = LLMStoryGenerator()
    
    def _load_lion_data(self) -> Dict:
        """بارگذاری داده‌های شیرها از فایل‌های موجود"""
        lion_data = {
            'Masai Mara, Kenya': {
                'coords': (-1.4066, 34.9066),
                'cost': 800, 'lion_density': 0.9,
                'description': 'Famous for big cat sightings and great migration',
                'lion_count': 15,
                'story_segment': 'In Masai Mara, we witness a pride of 15 lions resting under acacia trees. The dominant male watches over his family while cubs play nearby.'
            },
            'Serengeti, Tanzania': {
                'coords': (-2.3333, 34.8333),
                'cost': 900, 'lion_density': 0.85,
                'description': 'Vast plains with high lion population',
                'lion_count': 12,
                'story_segment': 'The Serengeti plains reveal a hunting party of 12 lions strategizing their approach to a wildebeest herd at sunset.'
            },
            'Kruger NP, South Africa': {
                'coords': (-24.0, 31.5),
                'cost': 700, 'lion_density': 0.8,
                'description': 'Excellent infrastructure and guaranteed sightings',
                'lion_count': 10,
                'story_segment': 'Kruger National Park offers close encounters with a pride of 10 lions, including impressive males with dark manes.'
            },
            'Okavango Delta, Botswana': {
                'coords': (-19.0, 23.0),
                'cost': 1200, 'lion_density': 0.75,
                'description': 'Unique water-based lion sightings',
                'lion_count': 8,
                'story_segment': 'In the Okavango Delta, we observe 8 lions adapting to the water-rich environment, a rare sight of swimming lions.'
            },
            'Samburu, Kenya': {
                'coords': (0.5255, 37.5914),
                'cost': 600, 'lion_density': 0.7,
                'description': 'Less crowded with unique wildlife',
                'lion_count': 6,
                'story_segment': 'Samburu reveals a smaller pride of 6 lions, including the rare desert-adapted lions with unique behaviors.'
            }
        }
        return lion_data
    
    def plan_budget_safari(self, start_location: str, budget: float) -> Dict:
        """برنامه‌ریزی سافاری بر اساس بودجه"""
        print(f"\n💰 Planning your ${budget:,} safari from {start_location}...")
        
        # بهینه‌سازی با PuLP
        model = pulp.LpProblem("Budget_Safari_Optimization", pulp.LpMaximize)
        locations = list(self.lion_locations.keys())
        
        # متغیرهای تصمیم
        x = pulp.LpVariable.dicts("visit", locations, cat='Binary')
        
        # تابع هدف: بیشینه کردن تعداد شیرهای مشاهده‌شده
        objective = pulp.lpSum([x[loc] * self.lion_locations[loc]['lion_count'] for loc in locations])
        model += objective
        
        # محدودیت بودجه
        model += pulp.lpSum([x[loc] * self.lion_locations[loc]['cost'] for loc in locations]) <= budget
        
        # محدودیت منطقی: حداکثر 4 مقصد
        model += pulp.lpSum([x[loc] for loc in locations]) <= 4
        
        # حل مدل
        model.solve()
        
        # استخراج نتایج
        selected_locations = [loc for loc in locations if x[loc].value() == 1]
        total_cost = sum([self.lion_locations[loc]['cost'] for loc in selected_locations])
        total_lions = sum([self.lion_locations[loc]['lion_count'] for loc in selected_locations])
        
        # ایجاد برنامه سفر
        itinerary = []
        for i, location in enumerate(selected_locations, 1):
            loc_data = self.lion_locations[location]
            itinerary.append({
                'day': i,
                'location': location,
                'country': location.split(', ')[1],
                'coordinates': loc_data['coords'],
                'cost': loc_data['cost'],
                'estimated_lions': loc_data['lion_count'],
                'description': loc_data['description'],
                'story_segment': loc_data.get('story_segment', 'An unforgettable wildlife experience.')
            })
        
        return {
            'selected_locations': selected_locations,
            'itinerary': itinerary,
            'total_cost': total_cost,
            'total_estimated_lions': total_lions,
            'budget_remaining': budget - total_cost,
            'optimization_status': pulp.LpStatus[model.status],
            'days_required': len(selected_locations)
        }

class InteractiveSafariMap:
    """نقشه تعاملی سافاری با تمام اطلاعات"""
    
    def __init__(self):
        self.base_map = None
    
    def create_comprehensive_map(self, safari_plan: Dict, original_data: List[Dict] = None) -> folium.Map:
        """ایجاد نقشه جامع با تمام اطلاعات"""
        print("🗺️ Creating comprehensive safari map...")
        
        # ایجاد نقشه پایه
        safari_map = folium.Map(
            location=[-8, 34],
            zoom_start=4,
            tiles='OpenStreetMap'
        )
        
        # اضافه کردن تمام داده‌های اصلی شیرها
        if original_data:
            self._add_original_lion_data(safari_map, original_data)
        
        # اضافه کردن مسیر بهینه‌شده
        self._add_optimized_route(safari_map, safari_plan)
        
        # اضافه کردن اطلاعات سفر
        self._add_journey_info(safari_map, safari_plan)
        
        return safari_map
    
    def _add_original_lion_data(self, map_obj: folium.Map, lion_data: List[Dict]):
        """اضافه کردن تمام داده‌های اصلی شیرها به نقشه"""
        for data in lion_data:
            try:
                if 'latitude' in data and 'longitude' in data:
                    folium.CircleMarker(
                        location=[data['latitude'], data['longitude']],
                        radius=6,
                        popup=f"""
                        🦁 Lion Sighting<br>
                        <b>Location:</b> {data.get('location', 'Unknown')}<br>
                        <b>Country:</b> {data.get('country', 'Unknown')}<br>
                        <b>Description:</b> {data.get('blip_description', 'No description')[:100]}...
                        """,
                        tooltip="Historical Lion Sighting",
                        color='orange',
                        fillColor='yellow',
                        fillOpacity=0.6
                    ).add_to(map_obj)
            except:
                continue
    
    def _add_optimized_route(self, map_obj: folium.Map, safari_plan: Dict):
        """اضافه کردن مسیر بهینه‌شده"""
        itinerary = safari_plan.get('itinerary', [])
        if len(itinerary) < 2:
            return
        
        # جمع‌آوری مختصات
        coordinates = []
        for stop in itinerary:
            coords = stop['coordinates']
            coordinates.append((coords[0], coords[1]))
            
            # اضافه کردن marker مقاصد بهینه‌شده
            folium.Marker(
                coords,
                popup=f"""
                🎯 Safari Destination<br>
                <b>{stop['location']}</b><br>
                Day {stop['day']}<br>
                Estimated Lions: {stop['estimated_lions']}<br>
                Cost: ${stop['cost']:,}
                """,
                tooltip=f"Day {stop['day']}: {stop['location']}",
                icon=folium.Icon(color='red', icon='star', prefix='fa')
            ).add_to(map_obj)
        
        # اضافه کردن خط مسیر
        folium.PolyLine(
            coordinates,
            color='blue',
            weight=4,
            opacity=0.7,
            popup=f"Optimized Safari Route - ${safari_plan['total_cost']:,}",
            tooltip="Your Safari Journey"
        ).add_to(map_obj)
        
        # اضافه کردن اعداد ترتیبی
        for i, stop in enumerate(itinerary):
            folium.Marker(
                stop['coordinates'],
                icon=DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(75,18),
                    html=f'<div style="font-size: 18pt; color: white; background: blue; border-radius: 50%; width: 30px; height: 30px; text-align: center; line-height: 30px;">{i+1}</div>'
                )
            ).add_to(map_obj)
    
    def _add_journey_info(self, map_obj: folium.Map, safari_plan: Dict):
        """اضافه کردن اطلاعات سفر به نقشه"""
        itinerary = safari_plan.get('itinerary', [])
        
        info_html = f"""
        <div style="position: fixed; top: 10px; left: 10px; background: white; 
        padding: 15px; border: 3px solid #27ae60; border-radius: 10px; z-index: 9999; 
        font-size: 12px; max-width: 300px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h3 style="margin: 0 0 10px 0; color: #2c3e50;">🦁 Your Lion Safari</h3>
        <b>Budget:</b> ${safari_plan.get('budget_used', 0):,}<br>
        <b>Destinations:</b> {len(itinerary)}<br>
        <b>Estimated Lions:</b> {safari_plan.get('total_estimated_lions', 0)}<br>
        <b>Days:</b> {len(itinerary)}<br>
        <b>Status:</b> {safari_plan.get('optimization_status', 'Optimized')}<br>
        <hr style="margin: 8px 0;">
        <b>Your Journey:</b><br>
        """
        
        for stop in itinerary:
            info_html += f"• Day {stop['day']}: {stop['location']}<br>"
        
        info_html += "</div>"
        map_obj.get_root().html.add_child(folium.Element(info_html))
        
        # اضافه کردن راهنما
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 10px; background: white; 
        padding: 10px; border: 2px solid grey; z-index: 9999; font-size: 11px;">
        <h4 style="margin: 0 0 5px 0;">Map Legend</h4>
        <p>🔴 <b>Star</b> - Your safari destinations</p>
        <p>🟠 <b>Circle</b> - Historical lion sightings</p>
        <p>🔵 <b>Blue Line</b> - Your optimized route</p>
        <p>🔢 <b>Numbers</b> - Journey sequence</p>
        <p>🟢 <b>Top Left</b> - Safari information</p>
        </div>
        """
        map_obj.get_root().html.add_child(folium.Element(legend_html))

def main():
    """اجرای اصلی برنامه"""
    print("🦁 AFRICAN LION SAFARI PLANNER & STORYTELLER")
    print("=" * 55)
    
    # ایجاد نمونه‌ها
    safari_planner = EnhancedSafariPlanner()
    map_creator = InteractiveSafariMap()
    
    try:
        # دریافت اطلاعات از کاربر
        print("\n🎯 Let's plan your dream lion safari!")
        
        print("\n📍 Select your starting location:")
        for i, location in enumerate(safari_planner.start_locations.keys(), 1):
            print(f"   {i}. {location}")
        
        start_choice = input("Enter choice (1-4): ").strip()
        start_options = list(safari_planner.start_locations.keys())
        start_location = start_options[int(start_choice)-1] if start_choice.isdigit() and 1 <= int(start_choice) <= 4 else start_options[0]
        
        print(f"\n💰 Enter your safari budget (USD):")
        print("   With $3,000 you can visit multiple prime lion territories!")
        budget = float(input("Your budget: $"))
        
        # برنامه‌ریزی سافاری
        safari_plan = safari_planner.plan_budget_safari(start_location, budget)
        
        # نمایش نتایج
        print("\n" + "=" * 60)
        print("🎉 YOUR OPTIMIZED LION SAFARI PLAN")
        print("=" * 60)
        
        print(f"\n📍 Starting from: {start_location}")
        print(f"💰 Budget: ${budget:,}")
        print(f"💵 Total cost: ${safari_plan['total_cost']:,}")
        print(f"💸 Remaining: ${safari_plan['budget_remaining']:,}")
        print(f"🦁 Estimated lions to see: {safari_plan['total_estimated_lions']}")
        print(f"📅 Days required: {safari_plan['days_required']}")
        
        print(f"\n🗺️ Your Safari Journey:")
        for stop in safari_plan['itinerary']:
            print(f"   Day {stop['day']}: {stop['location']}")
            print(f"      Estimated lions: {stop['estimated_lions']}")
            print(f"      Cost: ${stop['cost']:,}")
            print(f"      Description: {stop['description']}")
        
        # تحلیل بودجه
        budget_usage = (safari_plan['total_cost'] / budget) * 100
        if budget >= 3000:
            print(f"\n🎊 EXCELLENT! With ${budget:,}, you can visit {len(safari_plan['itinerary'])} prime lion territories!")
            print("   You'll see multiple prides and experience diverse ecosystems!")
        elif budget >= 2000:
            print(f"\n👍 GOOD! With ${budget:,}, you can visit {len(safari_plan['itinerary'])} excellent lion habitats!")
            print("   You'll have amazing lion encounters and great photographic opportunities!")
        else:
            print(f"\n💡 With ${budget:,}, focus on 1-2 key locations for the best lion experiences!")
        
        # تولید داستان سفر
        print(f"\n📖 Generating your safari story...")
        safari_story = safari_planner.story_generator.generate_safari_story(
            safari_plan['itinerary'], 
            budget, 
            safari_plan['total_estimated_lions']
        )
        
        # ایجاد نقشه
        print(f"\n🗺️ Creating your interactive safari map...")
        
        # داده‌های نمونه برای نمایش تاریخی (در عمل از فایل‌های واقعی استفاده کنید)
        sample_historical_data = [
            {
                'location': 'Masai Mara', 'country': 'Kenya',
                'latitude': -1.4066, 'longitude': 34.9066,
                'blip_description': 'Lion pride resting in savanna'
            },
            {
                'location': 'Serengeti', 'country': 'Tanzania', 
                'latitude': -2.3333, 'longitude': 34.8333,
                'blip_description': 'Male lion on lookout'
            }
        ]
        
        safari_map = map_creator.create_comprehensive_map(safari_plan, sample_historical_data)
        
        # ذخیره خروجی‌ها
        output_data = {
            'safari_plan': safari_plan,
            'safari_story': safari_story,
            'budget_analysis': {
                'original_budget': budget,
                'total_cost': safari_plan['total_cost'],
                'remaining_budget': safari_plan['budget_remaining'],
                'budget_usage_percent': budget_usage,
                'cost_per_lion': round(safari_plan['total_cost'] / safari_plan['total_estimated_lions'], 2) if safari_plan['total_estimated_lions'] > 0 else 0
            },
            'itinerary_details': safari_plan['itinerary'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'recommendations': [
                "Book lodges in advance during peak season",
                "Hire experienced local guides for best sightings",
                "Carry binoculars and good camera equipment",
                "Respect wildlife viewing distances",
                "Support local conservation efforts"
            ]
        }
        
        # ذخیره JSON
        with open('safari_adventure_complete.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # ذخیره نقشه
        safari_map.save('interactive_safari_adventure_map.html')
        
        # ذخیره داستان جداگانه
        with open('safari_story.txt', 'w', encoding='utf-8') as f:
            f.write(safari_story)
        
        print(f"\n✅ SUCCESS! Your safari adventure is ready!")
        print(f"📁 Files created:")
        print(f"   • interactive_safari_adventure_map.html - Interactive map with all data")
        print(f"   • safari_adventure_complete.json - Complete safari plan and story")
        print(f"   • safari_story.txt - Your personalized safari story")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Open 'interactive_safari_adventure_map.html' in your browser")
        print(f"   2. Read your safari story in 'safari_story.txt'")
        print(f"   3. Contact safari operators to book your adventure!")
        print(f"   4. With ${budget:,}, you'll see ~{safari_plan['total_estimated_lions']} lions across {len(safari_plan['itinerary'])} destinations!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()