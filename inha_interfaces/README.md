# inha_interfaces

## Overview
The `inha_interfaces` package provides an action server implementation for capturing face crops. It defines the necessary action messages and interfaces for interacting with the server.

## Action Definition
The package includes the action definition located in `action/CaptureFaceCrop.action`. This action allows users to capture images of a person's face and save them in a specified directory. The action message includes:

- **Goal**: 
  - `string person_name`: The name of the person, which will be used to create a directory for saving images.
  - `int32 num_images`: The number of images to capture.
  - `int32 timeout_ms`: The duration in milliseconds for which images will be captured.

- **Result**: 
  - `bool success`: Indicates whether the action was successful.
  - `string saved_dir`: The directory where the images are saved.
  - `string message`: A message providing details about the success or failure of the action.

- **Feedback**: 
  - `string state`: The current state of the action, which can be "WAITING_PERSON", "CAPTURING", "SAVING", etc.

## Build Instructions
To build the package, ensure that you have the necessary dependencies installed and then run the following commands in the root of the workspace:

```bash
catkin_make
```

## Usage
After building the package, you can launch the action server and send goals to capture face images. Make sure to provide the required parameters such as the person's name and the number of images to capture.

## Maintainers
- [Your Name] - [Your Email]

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.