import streamlit as st

from src.detect import process_image, detect_helmet


def main():
	st.title("Safety Detection")
	file = st.file_uploader('Upload Image ', type = ['jpg', 'png', 'jpeg'])
	if file is not None:
		image = process_image(file)
		annotated_image, has_helmet = detect_helmet(image = image)
		st.image(
			annotated_image, caption = "Processed Image", use_column_width = True
		)
		if has_helmet:
			st.success("Safety Helmet detected!")
		else:
			st.warning("No safety helmet detected!")


if __name__ == '__main__':
	main()
