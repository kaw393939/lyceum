// Visualization JavaScript for Lyceum
const visualizer = {
    /**
     * Render the home page visualization
     */
    renderHomeVisualization: function() {
        const container = document.getElementById('hero-visualization');
        if (!container) {
            console.error("Hero visualization container not found");
            return;
        }
        
        try {
            // Clear previous content
            container.innerHTML = '';
            
            // Create a force-directed graph using D3
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            // Sample data for knowledge graph
            const nodes = [
                { id: "lyceum", group: 1, label: "Lyceum", size: 30, description: "Educational platform blending ancient wisdom with AI" },
                { id: "philosophy", group: 2, label: "Philosophy", size: 20, description: "Foundational thinking for knowledge acquisition" },
                { id: "dialogue", group: 3, label: "Dialogue", size: 20, description: "Interactive learning through conversation" },
                { id: "ai", group: 4, label: "AI", size: 20, description: "Artificial intelligence technologies" },
                { id: "learning", group: 2, label: "Learning", size: 20, description: "Personalized educational journeys" },
                { id: "critical_thinking", group: 3, label: "Critical Thinking", size: 18, description: "Analytical reasoning skills" },
                { id: "personalization", group: 4, label: "Personalization", size: 18, description: "Adapting to individual learners" },
                { id: "knowledge", group: 2, label: "Knowledge", size: 18, description: "Interconnected concept mapping" },
                { id: "mentorship", group: 3, label: "Mentorship", size: 18, description: "Guided learning experiences" },
                { id: "socratic", group: 3, label: "Socratic Method", size: 18, description: "Learning through questioning" },
                { id: "content", group: 5, label: "Content", size: 18, description: "Adaptive educational materials" },
                { id: "assessment", group: 5, label: "Assessment", size: 18, description: "Measuring understanding and progress" }
            ];
            
            const links = [
                { source: "lyceum", target: "philosophy", value: 8 },
                { source: "lyceum", target: "dialogue", value: 8 },
                { source: "lyceum", target: "ai", value: 8 },
                { source: "lyceum", target: "learning", value: 8 },
                { source: "philosophy", target: "critical_thinking", value: 5 },
                { source: "dialogue", target: "socratic", value: 5 },
                { source: "ai", target: "personalization", value: 5 },
                { source: "learning", target: "knowledge", value: 5 },
                { source: "learning", target: "mentorship", value: 5 },
                { source: "dialogue", target: "mentorship", value: 3 },
                { source: "critical_thinking", target: "socratic", value: 3 },
                { source: "ai", target: "knowledge", value: 3 },
                { source: "philosophy", target: "socratic", value: 3 },
                { source: "lyceum", target: "content", value: 6 },
                { source: "content", target: "assessment", value: 4 },
                { source: "assessment", target: "personalization", value: 4 },
                { source: "learning", target: "assessment", value: 3 }
            ];
            
            // Create SVG element with responsive viewBox
            const svg = d3.select(container)
                .append("svg")
                .attr("width", "100%")
                .attr("height", "100%")
                .attr("viewBox", [0, 0, width, height])
                .attr("preserveAspectRatio", "xMidYMid meet");
            
            // Add definitions for gradients and glows
            const defs = svg.append("defs");
            
            // Add a subtle glow effect
            const glow = defs.append("filter")
                .attr("id", "glow")
                .attr("x", "-50%")
                .attr("y", "-50%")
                .attr("width", "200%")
                .attr("height", "200%");
                
            glow.append("feGaussianBlur")
                .attr("stdDeviation", "2")
                .attr("result", "coloredBlur");
                
            const feMerge = glow.append("feMerge");
            feMerge.append("feMergeNode")
                .attr("in", "coloredBlur");
            feMerge.append("feMergeNode")
                .attr("in", "SourceGraphic");
            
            // Create color gradients for nodes
            const blueGradient = defs.append("linearGradient")
                .attr("id", "blueGradient")
                .attr("x1", "0%")
                .attr("y1", "0%")
                .attr("x2", "100%")
                .attr("y2", "100%");
                
            blueGradient.append("stop")
                .attr("offset", "0%")
                .attr("stop-color", "#1a237e");
                
            blueGradient.append("stop")
                .attr("offset", "100%")
                .attr("stop-color", "#3949ab");
                
            // Create background
            svg.append("rect")
                .attr("width", width)
                .attr("height", height)
                .attr("fill", "none");
            
            // Create a force simulation
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(d => 100 + (20 - d.source.size) + (20 - d.target.size)))
                .force("charge", d3.forceManyBody().strength(-250))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collide", d3.forceCollide().radius(d => d.size + 15));
            
            // Create a tooltip
            const tooltip = d3.select(container)
                .append("div")
                .attr("class", "visualization-tooltip")
                .style("position", "absolute")
                .style("visibility", "hidden")
                .style("background-color", "rgba(26, 35, 126, 0.9)")
                .style("color", "white")
                .style("padding", "8px 12px")
                .style("border-radius", "6px")
                .style("font-size", "14px")
                .style("pointer-events", "none")
                .style("z-index", "10")
                .style("max-width", "200px")
                .style("box-shadow", "0 4px 8px rgba(0,0,0,0.2)")
                .style("transition", "opacity 0.2s ease");
            
            // Draw links with gradients
            const link = svg.append("g")
                .selectAll("line")
                .data(links)
                .enter()
                .append("line")
                .attr("stroke-width", d => Math.sqrt(d.value) * 0.8)
                .attr("stroke", d => {
                    // Use different colors based on the source node's group
                    const colors = ["rgba(26, 35, 126, 0.4)", "rgba(57, 73, 171, 0.4)", 
                                   "rgba(92, 107, 192, 0.4)", "rgba(121, 134, 203, 0.4)",
                                   "rgba(159, 168, 218, 0.4)"];
                    return colors[nodes.find(n => n.id === d.source.id || n.id === d.source).group % colors.length];
                })
                .attr("stroke-linecap", "round")
                .style("transition", "stroke-opacity 0.3s ease, stroke-width 0.3s ease");
            
            // Create a group for each node
            const node = svg.append("g")
                .selectAll(".node")
                .data(nodes)
                .enter()
                .append("g")
                .attr("class", "node")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            // Draw circles for nodes
            node.append("circle")
                .attr("r", d => d.size)
                .attr("fill", d => {
                    // Different shades of blue for different groups
                    const colors = ["url(#blueGradient)", "#283593", "#3949ab", "#5c6bc0", "#7986cb"];
                    return colors[d.group % colors.length];
                })
                .attr("stroke", "#fdd835")
                .attr("stroke-width", d => d.id === "lyceum" ? 3 : 2)
                .attr("filter", "url(#glow)")
                .style("cursor", "pointer")
                .style("transition", "r 0.3s ease, fill 0.3s ease, stroke-width 0.3s ease")
                .on("mouseover", function(event, d) {
                    // Highlight node on hover
                    d3.select(this)
                        .attr("r", d.size * 1.2)
                        .attr("stroke-width", d.id === "lyceum" ? 4 : 3);
                    
                    // Show tooltip
                    tooltip
                        .style("visibility", "visible")
                        .style("opacity", 1)
                        .html(`<strong>${d.label}</strong><br>${d.description}`);
                    
                    // Highlight connected links and nodes
                    link
                        .style("stroke-opacity", l => {
                            return l.source.id === d.id || l.target.id === d.id ? 1 : 0.1;
                        })
                        .style("stroke-width", l => {
                            return (l.source.id === d.id || l.target.id === d.id) 
                                ? Math.sqrt(l.value) * 1.5 : Math.sqrt(l.value) * 0.5;
                        });
                    
                    // Highlight connected nodes
                    node.selectAll("circle")
                        .style("opacity", n => {
                            return isConnected(d, n) ? 1 : 0.3;
                        });
                    
                    node.selectAll("text")
                        .style("opacity", n => {
                            return isConnected(d, n) ? 1 : 0.3;
                        });
                })
                .on("mousemove", function(event) {
                    // Position tooltip near the mouse
                    tooltip
                        .style("left", (event.pageX + 15) + "px")
                        .style("top", (event.pageY - 20) + "px");
                })
                .on("mouseout", function() {
                    // Reset node style
                    d3.select(this)
                        .attr("r", d => d.size)
                        .attr("stroke-width", d => d.id === "lyceum" ? 3 : 2);
                    
                    // Hide tooltip
                    tooltip
                        .style("visibility", "hidden")
                        .style("opacity", 0);
                    
                    // Reset all links and nodes
                    link
                        .style("stroke-opacity", 0.6)
                        .style("stroke-width", d => Math.sqrt(d.value) * 0.8);
                    
                    node.selectAll("circle").style("opacity", 1);
                    node.selectAll("text").style("opacity", 1);
                });
            
            // Add text labels to nodes
            node.append("text")
                .attr("dx", d => d.size + 5)
                .attr("dy", ".35em")
                .text(d => d.label)
                .attr("fill", "white")
                .attr("font-family", "var(--font-sans)")
                .attr("font-size", d => d.id === "lyceum" ? "16px" : "14px")
                .attr("font-weight", d => d.id === "lyceum" ? "bold" : "normal")
                .attr("stroke", "rgba(0,0,0,0.3)")
                .attr("stroke-width", 2)
                .attr("stroke-linejoin", "round")
                .attr("paint-order", "stroke")
                .style("pointer-events", "none");
            
            // Helper function to check if two nodes are connected
            function isConnected(a, b) {
                if (a.id === b.id) return true; // Same node
                return links.some(l => 
                    (l.source.id === a.id && l.target.id === b.id) || 
                    (l.source.id === b.id && l.target.id === a.id));
            }
            
            // Update positions on tick
            simulation.on("tick", () => {
                link
                    .attr("x1", d => Math.max(20, Math.min(width - 20, d.source.x)))
                    .attr("y1", d => Math.max(20, Math.min(height - 20, d.source.y)))
                    .attr("x2", d => Math.max(20, Math.min(width - 20, d.target.x)))
                    .attr("y2", d => Math.max(20, Math.min(height - 20, d.target.y)));
                
                node.attr("transform", d => `translate(${Math.max(20, Math.min(width - 20, d.x))},${Math.max(20, Math.min(height - 20, d.y))})`);
            });
            
            // Drag functions
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
            
            // Add pulsating animation for a magical feeling
            function pulseAnimation() {
                node.selectAll("circle")
                    .transition()
                    .duration(2000)
                    .attr("stroke-width", d => d.id === "lyceum" ? 4 : 3)
                    .transition()
                    .duration(2000)
                    .attr("stroke-width", d => d.id === "lyceum" ? 3 : 2)
                    .on("end", pulseAnimation);
            }
            
            // Start pulsating animation
            pulseAnimation();
                
            console.log("Home visualization rendered successfully");
        } catch (error) {
            console.error("Error rendering home visualization:", error);
            // Create a fallback visualization
            container.innerHTML = '<div class="visualization-placeholder">Knowledge Network Visualization</div>';
        }
    },
    
    /**
     * Render the home platform diagram
     */
    renderHomePlatformDiagram: function() {
        const container = document.getElementById('home-platform-diagram');
        if (!container) return;
        
        try {
            // Try to use mermaid.js for rendering
            const diagram = `
            graph TD
                lyceum[Lyceum Core]
                kg[Knowledge Graph]
                dialogue[Dialogue Engine]
                content[Content Generator]
                learning[Learning Paths]
                mentor[Mentor Service]
                
                lyceum --> kg
                lyceum --> dialogue
                lyceum --> content
                lyceum --> learning
                lyceum --> mentor
                
                kg -.-> dialogue
                kg -.-> content
                kg -.-> learning
                
                dialogue -.-> mentor
                learning -.-> mentor
                
                classDef core fill:#1a237e,stroke:#daa520,stroke-width:2px,color:white;
                classDef services fill:#3949ab,stroke:#daa520,stroke-width:2px,color:white;
                
                class lyceum core;
                class kg,dialogue,content,learning,mentor services;
            `;
            
            container.innerHTML = diagram;
            
            mermaid.initialize({
                securityLevel: 'loose',
                theme: 'dark',
                themeVariables: {
                    primaryColor: '#1a237e',
                    primaryTextColor: '#ffffff',
                    primaryBorderColor: '#daa520',
                    lineColor: '#3949ab',
                    secondaryColor: '#3949ab',
                    tertiaryColor: '#f5f5f5'
                }
            });
            
            mermaid.render('platform-diagram', diagram, (svgCode) => {
                container.innerHTML = svgCode;
            });
            
            console.log("Platform diagram rendered successfully");
        } catch (error) {
            console.error("Error rendering platform diagram:", error);
            // Create a fallback diagram
            container.innerHTML = `
                <div class="placeholder-diagram">
                    <div class="platform-node central">Lyceum Core</div>
                    <div class="platform-nodes">
                        <div class="platform-node">Knowledge Graph</div>
                        <div class="platform-node">Dialogue Engine</div>
                        <div class="platform-node">Content Generator</div>
                        <div class="platform-node">Learning Paths</div>
                        <div class="platform-node">Mentor Service</div>
                    </div>
                </div>
            `;
        }
    },
    
    /**
     * Render the technical page diagrams
     */
    renderTechnicalDiagram: function() {
        const container = document.getElementById('technical-diagram');
        if (!container) return;
        
        try {
            // Try to use mermaid.js for rendering
            const diagram = `
            graph TB
                user[User]
                ui[User Interface]
                api[API Gateway]
                
                kg[Knowledge Graph Service]
                dialogue[Dialogue Engine]
                content[Content Generator]
                paths[Learning Paths]
                mentor[Mentor Service]
                
                db1[(Neo4j)]
                db2[(MongoDB)]
                db3[(Vector DB)]
                
                user --> ui
                ui --> api
                api --> kg
                api --> dialogue
                api --> content
                api --> paths
                api --> mentor
                
                kg --> db1
                kg --> db3
                content --> db2
                dialogue --> kg
                paths --> kg
                mentor --> paths
                mentor --> dialogue
                
                classDef user fill:#009688,stroke:#004d40,stroke-width:2px,color:white;
                classDef interface fill:#26a69a,stroke:#004d40,stroke-width:2px,color:white;
                classDef service fill:#1a237e,stroke:#daa520,stroke-width:2px,color:white;
                classDef database fill:#283593,stroke:#daa520,stroke-width:2px,color:white;
                
                class user,ui user;
                class api interface;
                class kg,dialogue,content,paths,mentor service;
                class db1,db2,db3 database;
            `;
            
            container.innerHTML = diagram;
            
            mermaid.initialize({
                securityLevel: 'loose',
                theme: 'dark',
                themeVariables: {
                    primaryColor: '#1a237e',
                    primaryTextColor: '#ffffff',
                    primaryBorderColor: '#daa520',
                    lineColor: '#3949ab',
                    secondaryColor: '#3949ab',
                    tertiaryColor: '#f5f5f5'
                }
            });
            
            mermaid.render('technical-diagram-svg', diagram, (svgCode) => {
                container.innerHTML = svgCode;
            });
            
            console.log("Technical diagram rendered successfully");
        } catch (error) {
            console.error("Error rendering technical diagram:", error);
            // Create a fallback diagram
            container.innerHTML = `
                <div class="placeholder-diagram">
                    <div class="platform-node central">Lyceum Technical Architecture</div>
                    <div class="platform-nodes">
                        <div class="platform-node">User Interface</div>
                        <div class="platform-node">API Gateway</div>
                        <div class="platform-node">Microservices</div>
                        <div class="platform-node">Databases</div>
                    </div>
                </div>
            `;
        }
    },
    
    /**
     * Render agile roadmap for the agile page
     */
    renderAgileRoadmap: function() {
        const container = document.getElementById('roadmap-diagram');
        if (!container) return;
        
        try {
            // Try to use mermaid.js for rendering
            const diagram = `
            gantt
                title Lyceum Development Roadmap
                dateFormat  YYYY-MM-DD
                axisFormat %b %Y
                
                section Foundation
                Knowledge Graph MVP        :done, kg1, 2024-10-01, 90d
                Dialogue Engine v1         :done, de1, 2024-10-15, 75d
                Learning Paths Framework   :active, lp1, 2025-01-15, 90d
                
                section Features
                Content Generation         :active, cg1, 2025-01-01, 60d
                Mentor Service Beta        :active, ms1, 2025-02-15, 90d
                User Interface v1          :ui1, 2025-03-01, 60d
                Mobile Support             :ms2, after ui1, 45d
                
                section Integrations
                LMS Integrations           :lms, 2025-04-01, 60d
                Analytics Platform         :ap1, 2025-05-01, 45d
                API Partner Ecosystem      :api, 2025-06-01, 90d
            `;
            
            container.innerHTML = diagram;
            
            mermaid.initialize({
                securityLevel: 'loose',
                theme: 'default',
                gantt: {
                    titleTopMargin: 25,
                    barHeight: 30,
                    barGap: 6,
                    topPadding: 50,
                    leftPadding: 75,
                    gridLineStartPadding: 35,
                    fontSize: 12,
                    sectionFontSize: 14,
                    numberSectionStyles: 3,
                    axisFormat: '%b %Y',
                    todayMarker: 'rgba(26, 35, 126, 0.5)'
                }
            });
            
            mermaid.render('roadmap-diagram-svg', diagram, (svgCode) => {
                container.innerHTML = svgCode;
            });
            
            console.log("Agile roadmap rendered successfully");
        } catch (error) {
            console.error("Error rendering agile roadmap:", error);
            // Create a fallback diagram
            container.innerHTML = `
                <div class="placeholder-diagram">
                    <h3>Lyceum Development Roadmap</h3>
                    <div class="platform-node central">2025 Milestones</div>
                    <div class="platform-nodes">
                        <div class="platform-node">Q1: Learning Paths Framework</div>
                        <div class="platform-node">Q2: Mentor Service Launch</div>
                        <div class="platform-node">Q3: Mobile Support</div>
                        <div class="platform-node">Q4: Partner Integrations</div>
                    </div>
                </div>
            `;
        }
    }
};
