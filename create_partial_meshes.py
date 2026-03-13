# Create partial meshes to use for shape completion validation against original/ground truth meshes
# Subtract segment PLY files generated using ATLAS from original/ground truth meshes

import os
import vtk
import json
import argparse
from glob import glob
import pyvista as pv
from scipy.spatial import cKDTree

def save_ply(polydata, path, binary=True):
    if not isinstance(polydata, pv.PolyData):
        polydata = pv.wrap(polydata)
    polydata.save(path, binary=binary)

def clean_triangulate(pd):
    cl = vtk.vtkCleanPolyData()
    cl.SetInputData(pd); cl.ConvertStripsToPolysOn(); cl.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cl.GetOutputPort()); tri.PassLinesOff(); tri.PassVertsOff(); tri.Update()
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputConnection(tri.GetOutputPort())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    
    return pv.wrap(conn.GetOutput())

def fast_subtract(original_pd, segment_pd, eps=0.02):
    # Delete faces near segment to remove
    orig = pv.wrap(original_pd)
    seg = pv.wrap(segment_pd)
    # Get face centers
    orig.compute_cell_sizes(length=False, area=False, volume=False)
    face_centers = orig.cell_centers().points
    # Find faces near segment
    tree = cKDTree(seg.points)
    dists, _ = tree.query(face_centers, k=1)
    # Keep faces far from segment
    orig['keep_face'] = (dists > eps).astype(int)
    result = orig.threshold(0.5, scalars='keep_face', invert=False)
    return result.extract_surface().clean().triangulate()

def flip_to_ras(polydata):
    # LPS to RAS coords (x flip)
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(polydata)
    tf.SetTransform(transform)
    tf.Update()
    return tf.GetOutput()

def create_partial_mesh(original_ply, segment_ply, output_ply):
    original = pv.wrap(clean_triangulate(pv.read(original_ply)))    
    segment = pv.wrap(clean_triangulate(pv.read(segment_ply)))
    segment = pv.wrap(flip_to_ras(segment))
    # Debug: colored mesh by pt distances
    #original_with_dist = original.compute_implicit_distance(segment, inplace=False)
    #original_with_dist.save(output_ply.replace(".ply", "_distances.vtp"))
    partial = fast_subtract(original, segment, eps=0.01)
    print(f"  Original: {original.GetNumberOfPoints()} vertices")
    print(f"  Segment:  {segment.GetNumberOfPoints()} vertices")
    print(f"  Partial:  {partial.GetNumberOfPoints()} vertices")
    save_ply(partial, output_ply)
    print(f"  ✓ Saved: {output_ply}")
    return partial.GetNumberOfPoints()

def create_validation_dataset(original_dir, segments_dir, output_dir, segment_to_remove=5):
    # Create output directory for partial meshes
    partial_dir = os.path.join(output_dir, "partial_meshes")
    os.makedirs(partial_dir, exist_ok=True)
    # Find all segment files for the target segment
    segment_pattern = os.path.join(segments_dir, f"*_seg_{segment_to_remove:02d}.ply")
    segment_files = glob(segment_pattern)
    if not segment_files:
        raise RuntimeError(f"No segment files found matching: {segment_pattern}")
    print(f"Found {len(segment_files)} meshes with segment {segment_to_remove}")
    results = []
    for segment_file in segment_files:
        # Extract base name
        basename = os.path.basename(segment_file)
        base_name = basename.replace(f"_seg_{segment_to_remove:02d}.ply", "")
        print(f"\n{'='*60}")
        # Find corresponding original mesh (ground truth)
        original_ply = os.path.join(original_dir, f"{base_name}.ply")
        if not os.path.exists(original_ply):
            vtk_path = os.path.join(original_dir, f"{base_name}.vtk")
            if os.path.exists(vtk_path):
                original_ply = vtk_path
            else:
                print(f"Original mesh not found: {base_name}.ply or .vtk")
                continue

        try:
            # Create partial mesh
            partial_path = os.path.join(partial_dir, f"{base_name}_partial.ply")
            n_vertices = create_partial_mesh(original_ply, segment_file, partial_path) 
            results.append({
                'base_name': base_name,
                'ground_truth': original_ply,  # Just reference the original
                'partial': partial_path,
                'removed_segment': segment_file,  # Just reference existing segment
                'partial_vertices': int(n_vertices)
            })
            print(f"  ✓ Success!")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    # Save summary
    summary = {
        'segment_removed': segment_to_remove,
        'n_meshes': len(results),
        'original_dir': os.path.abspath(original_dir),
        'segments_dir': os.path.abspath(segments_dir),
        'meshes': results
    }
    summary_path = os.path.join(output_dir, "partial_meshing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*70}")
    print(f"✓ Created {len(results)} partial meshes")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"\nDirectories:")
    print(f"  - Partial meshes (input to shape completion): {partial_dir}")
    print(f"  - Ground truth (compare with output):        {original_dir}")
    print(f"  - Removed segments (for reference):          {segments_dir}")
    print(f"{'='*70}")
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Create partial meshes for shape completion validation")
    parser.add_argument("original_dir", 
                       help="Directory with original meshes (ground truth)")
    parser.add_argument("segments_dir",
                       help="Directory with segment PLYs (*_seg_XX.ply)")
    parser.add_argument("output_dir",
                       help="Output directory for partial meshes")
    parser.add_argument("--segment", type=int, default=6,
                       help="Segment number to remove (default: 6)")
    args = parser.parse_args()
    create_validation_dataset(
        args.original_dir,
        args.segments_dir,
        args.output_dir,
        segment_to_remove=args.segment)

if __name__ == "__main__":
    main()