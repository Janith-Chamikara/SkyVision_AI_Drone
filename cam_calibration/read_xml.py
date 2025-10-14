import xml.etree.ElementTree as ET
import numpy as np

class CameraCalibration:
    """Class to store camera calibration parameters"""
    def __init__(self):
        self.metadata = {}
        self.focal_length = None
        self.focal_length_error = None
        self.principal_point = None
        self.principal_point_error = None
        self.radial_distortion = None
        self.radial_distortion_error = None
        self.tangential_distortion = None
        self.skew = None
        self.intrinsic_matrix = None
        self.rotation_vectors = []
        self.rotation_vectors_error = []
        self.translation_vectors = []
        self.translation_vectors_error = []
        self.mean_reprojection_error = None
        self.max_reprojection_error = None
        self.image_errors = []
    
    def __repr__(self):
        return (f"CameraCalibration(\n"
                f"  Focal Length: {self.focal_length}\n"
                f"  Principal Point: {self.principal_point}\n"
                f"  Radial Distortion: {self.radial_distortion}\n"
                f"  Mean Reprojection Error: {self.mean_reprojection_error}\n"
                f"  Num Patterns: {len(self.rotation_vectors)}\n"
                f")")

def load_camera_calibration(xml_file):
    """
    Load camera calibration parameters from XML file
    
    Args:
        xml_file (str): Path to XML file
        
    Returns:
        CameraCalibration: Object containing all calibration parameters
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    calib = CameraCalibration()
    
    # Parse metadata
    metadata = root.find('Metadata')
    if metadata is not None:
        for child in metadata:
            if child.tag == 'ImageSize':
                calib.metadata[child.tag] = [int(x) for x in child.text.split()]
            else:
                calib.metadata[child.tag] = child.text
    
    # Parse intrinsic parameters
    intrinsics = root.find('IntrinsicParameters')
    if intrinsics is not None:
        # Focal length
        focal = intrinsics.find('FocalLength')
        if focal is not None:
            calib.focal_length = np.array([
                float(focal.find('fx').text),
                float(focal.find('fy').text)
            ])
            calib.focal_length_error = np.array([
                float(focal.find('fx_error').text),
                float(focal.find('fy_error').text)
            ])
        
        # Principal point
        pp = intrinsics.find('PrincipalPoint')
        if pp is not None:
            calib.principal_point = np.array([
                float(pp.find('cx').text),
                float(pp.find('cy').text)
            ])
            calib.principal_point_error = np.array([
                float(pp.find('cx_error').text),
                float(pp.find('cy_error').text)
            ])
        
        # Radial distortion
        radial = intrinsics.find('RadialDistortion')
        if radial is not None:
            k_coeffs = []
            k_errors = []
            i = 1
            while radial.find(f'k{i}') is not None:
                k_coeffs.append(float(radial.find(f'k{i}').text))
                k_errors.append(float(radial.find(f'k{i}_error').text))
                i += 1
            calib.radial_distortion = np.array(k_coeffs)
            calib.radial_distortion_error = np.array(k_errors)
        
        # Tangential distortion
        tangential = intrinsics.find('TangentialDistortion')
        if tangential is not None:
            calib.tangential_distortion = np.array([
                float(tangential.find('p1').text),
                float(tangential.find('p2').text)
            ])
        
        # Skew
        skew = intrinsics.find('Skew')
        if skew is not None:
            calib.skew = float(skew.text)
        
        # Intrinsic matrix
        K = intrinsics.find('IntrinsicMatrix')
        if K is not None:
            matrix_rows = []
            for i in range(1, 4):
                row = K.find(f'row{i}')
                if row is not None:
                    matrix_rows.append([float(x) for x in row.text.split()])
            calib.intrinsic_matrix = np.array(matrix_rows)
    
    # Parse extrinsic parameters
    extrinsics = root.find('ExtrinsicParameters')
    if extrinsics is not None:
        for pattern in extrinsics.findall('Pattern'):
            # Rotation vector
            rvec = pattern.find('RotationVector')
            if rvec is not None:
                calib.rotation_vectors.append(np.array([
                    float(rvec.find('rx').text),
                    float(rvec.find('ry').text),
                    float(rvec.find('rz').text)
                ]))
                calib.rotation_vectors_error.append(np.array([
                    float(rvec.find('rx_error').text),
                    float(rvec.find('ry_error').text),
                    float(rvec.find('rz_error').text)
                ]))
            
            # Translation vector
            tvec = pattern.find('TranslationVector')
            if tvec is not None:
                calib.translation_vectors.append(np.array([
                    float(tvec.find('tx').text),
                    float(tvec.find('ty').text),
                    float(tvec.find('tz').text)
                ]))
                calib.translation_vectors_error.append(np.array([
                    float(tvec.find('tx_error').text),
                    float(tvec.find('ty_error').text),
                    float(tvec.find('tz_error').text)
                ]))
        
        # Convert lists to arrays
        if calib.rotation_vectors:
            calib.rotation_vectors = np.array(calib.rotation_vectors)
            calib.rotation_vectors_error = np.array(calib.rotation_vectors_error)
            calib.translation_vectors = np.array(calib.translation_vectors)
            calib.translation_vectors_error = np.array(calib.translation_vectors_error)
    
    # Parse reprojection errors
    reproj = root.find('ReprojectionErrors')
    if reproj is not None:
        mean_err = reproj.find('MeanError')
        if mean_err is not None:
            calib.mean_reprojection_error = float(mean_err.text)
        
        max_err = reproj.find('MaxError')
        if max_err is not None:
            calib.max_reprojection_error = float(max_err.text)
        
        for img_err in reproj.findall('ImageError'):
            calib.image_errors.append(float(img_err.text))
        
        if calib.image_errors:
            calib.image_errors = np.array(calib.image_errors)
    
    return calib


# Example usage
if __name__ == "__main__":
    # Load calibration
    calib = load_camera_calibration('camera_calibration.xml')
    
    # Print summary
    print("=" * 60)
    print("Camera Calibration Parameters")
    print("=" * 60)
    print(f"\nMetadata:")
    for key, value in calib.metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\nIntrinsic Parameters:")
    print(f"  Focal Length (pixels): {calib.focal_length.fx}")
    print(f"  Focal Length Error: ±{calib.focal_length_error}")
    print(f"  Principal Point (pixels): {calib.principal_point}")
    print(f"  Principal Point Error: ±{calib.principal_point_error}")
    print(f"  Radial Distortion: {calib.radial_distortion}")
    print(f"  Radial Distortion Error: ±{calib.radial_distortion_error}")
    
    print(f"\nIntrinsic Matrix (K):")
    print(calib.intrinsic_matrix)
    
    print(f"\nReprojection Errors:")
    print(f"  Mean: {calib.mean_reprojection_error:.4f} pixels")
    print(f"  Max: {calib.max_reprojection_error:.4f} pixels")
    
    print(f"\nExtrinsic Parameters:")
    print(f"  Number of patterns: {len(calib.rotation_vectors)}")
    print(f"  Rotation vectors shape: {calib.rotation_vectors.shape}")
    print(f"  Translation vectors shape: {calib.translation_vectors.shape}")