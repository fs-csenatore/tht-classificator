import setuptools

setuptools.setup(
 	name="THT-Classificator",
 	version="0.1.0",
	python_requires=">=3.9",
    packages=setuptools.find_namespace_packages(where="src"),
    package_dir={"": "src"},
	package_data={
        "THTClassificator.bin": ["*.pkl", "*.tflite"],
        "THTClassificator.SettingsFile": ["*.xml"],
        },
    include_package_data=True,
 	install_requires=[
 		'opencv-python >=4.7.0',
 		'numpy',
 		'dataclasses',
 		'sshkeyboard',
 	],
 	entry_points={
		'console_scripts': ['THT-Classificator=THTClassificator.__main__:main',]
	},
)