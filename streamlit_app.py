import streamlit as st
import google.generativeai as genai
import ezdxf
from io import BytesIO
import matplotlib.pyplot as plt

# Configure the API key securely from Streamlit's secrets
# Make sure to add GOOGLE_API_KEY in secrets.toml (for local) or Streamlit Cloud Secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Text-to-CAD Design Application")
st.write("Use generative AI to create CAD designs based on your prompt.")

# Prompt input field
prompt = st.text_input("Enter your design prompt:", "Draw a simple house")

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
        
        # Create a new DXF document
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        # Example CAD design based on response (replace with actual logic)
        # This generates a simple square as a placeholder
        msp.add_line((0, 0), (10, 0))
        msp.add_line((10, 0), (10, 10))
        msp.add_line((10, 10), (0, 10))
        msp.add_line((0, 10), (0, 0))

        # Save the DXF document to a BytesIO object
        dxf_stream = BytesIO()
        doc.write_stream(dxf_stream)
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
        for e in msp.query('LINE'):
            start, end = e.dxf.start, e.dxf.end
            ax.plot([start.x, end.x], [start.y, end.y], 'k-')
        
        ax.set_aspect('equal')
        plt.grid(True)
        plt.title('CAD Design Visualization')
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        
        st.image(buf, caption="Generated CAD Design")
    except Exception as e:
        st.error(f"Error: {e}")
