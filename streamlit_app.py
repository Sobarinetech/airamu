import streamlit as st
import google.generativeai as genai
import ezdxf
from io import BytesIO
import matplotlib.pyplot as plt
import re

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Text-to-CAD Design Application")
st.write("Use generative AI to create CAD designs based on your prompt.")

# Prompt input field
prompt = st.text_input("Enter your design prompt:", "Design a simple car")

# Button to generate response
if st.button("Generate CAD Design"):
    try:
        # Load and configure the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response from the model
        response = model.generate_content(prompt)
        
        # Display response in Streamlit
        st.write("CAD Design Description:")
        st.write(response.text)
        
        # Function to parse description and create CAD elements
        def create_cad_elements(description):
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()

            # Dynamic parsing and creation of CAD elements
            commands = description.split('\n')
            for command in commands:
                command = command.strip().lower()
                if "rectangle" in command:
                    match = re.search(r'rectangle with width (\d+) and height (\d+)', command)
                    if match:
                        width, height = int(match.group(1)), int(match.group(2))
                        msp.add_lwpolyline([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
                elif "circle" in command:
                    match = re.search(r'circle with radius (\d+)', command)
                    if match:
                        radius = int(match.group(1))
                        msp.add_circle((0, 0), radius)
                elif "line" in command:
                    match = re.search(r'line from \((\d+), (\d+)\) to \((\d+), (\d+)\)', command)
                    if match:
                        x1, y1, x2, y2 = map(int, match.groups())
                        msp.add_line((x1, y1), (x2, y2))
                # Extend to handle more shapes as needed

            return doc

        # Create CAD elements based on the AI-generated description
        doc = create_cad_elements(response.text)
        
        # Save the DXF document to a BytesIO object
        dxf_stream = BytesIO()
        doc.saveas(dxf_stream)
        dxf_stream.seek(0)

        # Provide download link for the DXF file
        st.download_button(
            label="Download CAD Design",
            data=dxf_stream,
            file_name="design.dxf",
            mime="application/dxf"
        )

        # Visualize the CAD design (optional)
        fig, ax = plt.subplots()
        for e in doc.modelspace().query('LINE'):
            start, end = e.dxf.start, e.dxf.end
            ax.plot([start.x, end.x], [start.y, end.y], 'k-')
        for e in doc.modelspace().query('CIRCLE'):
            center, radius = e.dxf.center, e.dxf.radius
            circle = plt.Circle((center.x, center.y), radius, color='k', fill=False)
            ax.add_artist(circle)
        for e in doc.modelspace().query('LWPOLYLINE'):
            points = e.get_points()
            xs, ys = zip(*[(p[0], p[1]) for p in points])
            ax.plot(xs, ys, 'k-')
        
        ax.set_aspect('equal')
        plt.grid(True)
        plt.title('CAD Design Visualization')
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        
        st.image(buf, caption="Generated CAD Design")
    except Exception as e:
        st.error(f"Error: {e}")
