# Joyce Weather: A Live Literary Installation for Dublin

---

# Concept

A weather-driven interactive visualisation that maps James Joyce's four major works onto Dublin's daily weather conditions. The city's live forecast determines which Joyce text the viewer experiences and how it behaves on screen.

# Technical Approach

- Weather data: OpenWeatherMap API or Met Éireann API
- Frontend: HTML/JS, Pretext for text layout and measurement
- Physics: Matter.js for wind and rain particle simulation
- Text source: Project Gutenberg editions of Dubliners, Portrait, Ulysses (public domain). Finnegans Wake txt. 
- Update frequency: Weather check every 15–30 minutes, smooth transitions between states
- Deployment: GitHub Pages or standalone web app

# Pretext Integration

Pretext provides the text layout engine that makes all four weather states performant and precise. One measurement pipeline, four rendering behaviours.

#Core Advantage

Pretext calculates text height and width without DOM reads in the hot path. This is critical when text is constantly moving (falling as rain, blowing as wind, fading in darkness). Traditional DOM-based layout would choke on the continuous reflow. Pretext pre-computes everything, so the animation loop only has to position elements, never measure them. see [Pretext](https://chenglou.me/pretext/)
