# -*- coding: utf-8 -*-
import os
os.environ['abaqus_no_journal'] = '1'

from abaqus import *
from abaqusConstants import *
from regionToolset import Region
import numpy as np

import random

def func(dir,inp_name,model_name,part_name,assembly_name,element_size,thickness,thickness_layers,disp,cutoff,num_repeats):

    # Define parameters
    txt_file_path = dir+'/{}/img.txt'.format(inp_name)  # Replace with your txt file path
    model_name = model_name
    part_name = part_name
    assembly_name = assembly_name
    element_sizey,element_sizex = element_size[0],element_size[1]  # Size of the square element in the yx plane
    thickness = thickness  # Thickness of the solid elements in the z direction
    thickness_layers = thickness_layers  # Number of mesh layers in the thickness direction
    cutoffy,cutoffx = cutoff[0],cutoff[1]

    # Load the binary matrix from the txt file
    try:
        matrix = np.loadtxt(txt_file_path, dtype=int)[:cutoffy, :cutoffx]
    except Exception as e:
        print("Error reading the txt file:", e)
        exit()

    # Get matrix dimensions
    rows, cols = matrix.shape

    # Create a new model
    if model_name in mdb.models:
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    # Create a new part
    part = model.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)

    # Loop through rows in the matrix to create rectangles
    index = 1
    for i in range(rows):
        j = 0
        while j < cols:
            # Find the start of a sequence of 1's
            if matrix[i, j] == 1:
                start_j = j
                # Find the end of this sequence of 1's
                while j < cols and matrix[i, j] == 1:
                    j += 1
                end_j = j  # The column after the last 1 in the sequence
                
                # Calculate the coordinates of the rectangle
                x1 = start_j * element_sizex
                y1 = (rows - 1 - i) * element_sizey  # Flip the y-axis to match matrix orientation
                x2 = end_j * element_sizex
                y2 = y1 + element_sizey
                
                # Create a new sketch for the rectangle
                sketch = model.ConstrainedSketch(name='__profile__', sheetSize=200.0)
                sketch.rectangle(point1=(x1, y1), point2=(x2, y2))
                
                # Create a solid extrusion from the sketch
                part.BaseSolidExtrude(sketch=sketch, depth=thickness)
                index += 1
                # print('Rectangle {} created at row {}, columns {}-{}'.format(index, i, start_j, end_j-1))
                
                # Delete the sketch to save memory
                del model.sketches['__profile__']
            else:
                # Skip over zeros
                j += 1

    e = part.edges
    part.MergeEdges(edgeList = e, extendSelection=True)
    # Mirror the part along the x-axis
    x_axis = part.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=cutoffx * element_sizex)
    part.Mirror(mirrorPlane=part.datum[x_axis.id], keepOriginal=True)

    # Mirror the part along the y-axis
    y_axis = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0.0)
    part.Mirror(mirrorPlane=part.datum[y_axis.id], keepOriginal=True)

    # Create an assembly and instance the part
    assembly = model.rootAssembly
    instance = assembly.Instance(name=assembly_name, part=part, dependent=ON)

    num_repeats_x,num_repeats_y=num_repeats[0],num_repeats[1]
    # Generate array of instances in x and y directions
    for j in range(num_repeats_x):
        for i in range(num_repeats_y):
            if i == 0 and j == 0:
                # Skip the original instance at (0, 0)
                continue
            instance_name = 'BinaryPatternInstance_{}_{}'.format(i, j)
            assembly.Instance(
                name=instance_name,
                part=part,
                dependent=ON
            )
            # Translate instance to its new location
            assembly.translate(
                instanceList=(instance_name,),
                vector=(i * cutoffx*element_sizex*2, j * cutoffy*element_sizey*2, 0.0)
            )

    # Check the number of instances
    num_instances = len(assembly.instances)
    print("Number of instances in assembly:", num_instances)

    # Perform Boolean merge if there are at least two instances
    if num_instances >= 2:
        final_part_name = assembly_name
        assembly.InstanceFromBooleanMerge(
            name=final_part_name,
            instances=tuple(assembly.instances.values()),
            keepIntersections=ON,
            originalInstances=DELETE,
            domain=GEOMETRY
        )
        # 从模型中获取合并后的 part
        if final_part_name in model.parts:
            part = model.parts[final_part_name]
            print("Boolean merge successful. Merged part: {}".format(final_part_name))
            # 获取新的实例 (布尔合并后的实例名称)
            instance_name = final_part_name + '-1'
            if instance_name in assembly.instances:
                instance = assembly.instances[instance_name]
                print("Using merged instance: {}".format(instance_name))
            else:
                print("Error: Merged instance {} not found in assembly.".format(instance_name))
                exit()
        else:
            print("Error: Boolean merge failed. Final part not created.")
            exit()
    else:
        print("Only one instance found. Skipping Boolean merge.")
        # 如果未进行布尔合并，使用原始的 part 和实例
        part = model.parts[part_name]
        instance = assembly.instances[assembly_name]
        print("Using original part and instance: {}".format(part_name))

    # 确保实例更新
    assembly.regenerate()

    # Mesh the part
    # 1. Seed the part for the xy plane
    part.seedPart(size=(element_sizex+element_sizey)/2.0 * 2.0, deviationFactor=0.1, minSizeFactor=0.1)

    # 2. Seed the edges in the thickness direction
    thickness_edges = []
    for edge in part.edges:
        # Get the coordinates of the edge's two endpoints
        vertex_indices = edge.getVertices()
        point1 = part.vertices[vertex_indices[0]].pointOn[0]  # First endpoint coordinates
        point2 = part.vertices[vertex_indices[1]].pointOn[0]  # Second endpoint coordinates

        # Check if it's an edge along the thickness direction
        if abs(point1[0] - point2[0]) < 1e-6 and abs(point1[1] - point2[1]) < 1e-6 and abs(point1[2] - point2[2]) > 1e-6:
            thickness_edges.append(edge)

    # Seed edges in the thickness direction
    part.seedEdgeByNumber(edges=thickness_edges, number=thickness_layers, constraint=FINER)

    # Generate the mesh
    part.generateMesh()

    if len(part.elements) == 0:
        part.deleteMesh()
        part.deleteSeeds()
        # Mesh the part
        # 1. Seed the part for the xy plane
        part.seedPart(size=(element_sizex+element_sizey)/2.0 * 1.0, deviationFactor=0.1, minSizeFactor=0.1)

        # 2. Seed the edges in the thickness direction
        thickness_edges = []
        for edge in part.edges:
            # Get the coordinates of the edge's two endpoints
            vertex_indices = edge.getVertices()
            point1 = part.vertices[vertex_indices[0]].pointOn[0]  # First endpoint coordinates
            point2 = part.vertices[vertex_indices[1]].pointOn[0]  # Second endpoint coordinates

            # Check if it's an edge along the thickness direction
            if abs(point1[0] - point2[0]) < 1e-6 and abs(point1[1] - point2[1]) < 1e-6 and abs(point1[2] - point2[2]) > 1e-6:
                thickness_edges.append(edge)

        # Seed edges in the thickness direction
        part.seedEdgeByNumber(edges=thickness_edges, number=thickness_layers, constraint=FINER)
        # Generate the mesh
        part.generateMesh()


    if num_instances >= 2:
        instance = assembly.instances[instance_name]
    else:
        instance = assembly.instances[assembly_name]

    # Regenerate the assembly to update the instance
    assembly.regenerate()

    try:
        # Create node sets for the top and bottom surfaces (in Y direction)
        x_max = max(node.coordinates[0] for node in instance.nodes)
        x_min = min(node.coordinates[0] for node in instance.nodes)
        y_max = max(node.coordinates[1] for node in instance.nodes)
        y_min = min(node.coordinates[1] for node in instance.nodes)
        #  (x_max, 0, thickness)
        rightmost_nodes = instance.nodes.getByBoundingBox(
            xMin=x_max - 1e-6,
            xMax=x_max + 1e-6,
            yMin=-1e-6,
            yMax=1e-6,
            zMin=thickness - 1e-6,  # 直接指定Z=1
            zMax=thickness + 1e-6
        )
        if rightmost_nodes:
            assembly.Set(name='RightmostNode', nodes=rightmost_nodes[0:1])

    except ValueError as e:
        return

    top_nodes = instance.nodes.getByBoundingBox(yMin=y_max - 1e-6, yMax=y_max + 1e-6)
    bottom_nodes = instance.nodes.getByBoundingBox(yMin=y_min - 1e-6, yMax=y_min + 1e-6)

    assembly.Set(name='TopNodes', nodes=top_nodes)
    assembly.Set(name='BottomNodes', nodes=bottom_nodes)

    # Create reference points in the assembly
    top_rp = assembly.ReferencePoint(point=((x_max-x_min)/2, y_max, thickness / 2))
    bottom_rp = assembly.ReferencePoint(point=((x_max-x_min)/2, y_min, thickness / 2))

    # Define regions for reference points
    top_rp_region = Region(referencePoints=(assembly.referencePoints[top_rp.id], ))
    bottom_rp_region = Region(referencePoints=(assembly.referencePoints[bottom_rp.id], ))

    # Create coupling constraints in the assembly
    model.Coupling(
        name='TopCoupling',
        controlPoint=top_rp_region,
        surface=assembly.sets['TopNodes'],
        influenceRadius=WHOLE_SURFACE,
        couplingType=KINEMATIC,
        u1=ON, u2=ON, u3=ON,
        ur1=ON, ur2=ON, ur3=ON
    )
    model.Coupling(
        name='BottomCoupling',
        controlPoint=bottom_rp_region,
        surface=assembly.sets['BottomNodes'],
        influenceRadius=WHOLE_SURFACE,
        couplingType=KINEMATIC,
        u1=ON, u2=ON, u3=ON,
        ur1=ON, ur2=ON, ur3=ON
    )

    # Create geometric sets for the reference points
    assembly.Set(name='RP_Top', referencePoints=(assembly.referencePoints[top_rp.id], ))
    assembly.Set(name='RP_Bottom', referencePoints=(assembly.referencePoints[bottom_rp.id], ))

    # Define material properties
    material = model.Material(name="Steel")
    material.Elastic(table=((210000.0, 0.3), ))  # E=210000 MPa, v=0.3
    material.Plastic(table=((235.0, 0.0), ))  # Yield strength = 235 MPa

    # Create section and assign it to the part
    section = model.HomogeneousSolidSection(name="SteelSection", material="Steel", thickness=None)
    part.SectionAssignment(region=Region(cells=part.cells), sectionName="SteelSection")

    # Create analysis step
    model.StaticStep(name='Step1', previous='Initial', nlgeom=ON, description='Step to apply x-direction load on RP_Top')
    model.steps['Step1'].setValues(maxNumInc=500, initialInc=0.01, minInc=1e-10, maxInc=0.1, timePeriod=1.0)

    # Disable history output (turn off increment history recording)
    if 'H-Output-1' in model.historyOutputRequests.keys():
        del model.historyOutputRequests['H-Output-1']

    # Delete default field output request F-Output-1 (if exists)
    if 'F-Output-1' in model.fieldOutputRequests.keys():
        del model.fieldOutputRequests['F-Output-1']

    # Create field output request to capture only RP_Top displacement (U) and reaction force (RF)
    model.FieldOutputRequest(
        name='RP_Top_Output',  # Output request name
        createStepName='Step1',  # Created in Step1
        variables=('U', 'RF'),  # Output displacement (U) and reaction force (RF)
        region=assembly.sets['RP_Top'],  # Only for RP_Top
        numIntervals=64  # Set evenly spaced time intervals
    )

    model.FieldOutputRequest(
        name='RightmostNode_Output',
        createStepName='Step1',
        variables=('U',),
        region=assembly.sets['RightmostNode'],
        numIntervals=64
    )

    # Define specific field output request to record stresses and displacements at times 1, 2, 3, 4
    selected_times = [1.0, 2.0, 3.0, 4.0]  # Define interested time points
    model.FieldOutputRequest(
        name='MisesAndDisplacement',  # Output request name
        createStepName='Step1',  # Created in Step1
        variables=('S', 'U'),  # Output stress (S) and displacement (U)
        timeInterval=1.0  # Record every 1 unit of time
    )

    # Define load table
    # Format: amplitude_table = [(time1, amplitude1), (time2, amplitude2), ...]
    amplitude_table = [(0.0, 0.0), (1.0, disp), (2.0, 0.0), (3.0, -disp), (4.0, 0.0)]

    # Create the amplitude for the load
    model.TabularAmplitude(name='LoadAmplitude', timeSpan=STEP, data=amplitude_table)

    # Apply boundary conditions
    # Fix bottom RP in all degrees of freedom
    model.DisplacementBC(name='FixBottomRP', createStepName='Initial',
                        region=assembly.sets['RP_Bottom'], 
                        u1=0.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)

    # Apply load to top RP using the amplitude
    # model.DisplacementBC(name='LoadTopRP', createStepName='Step1',
    #                     region=assembly.sets['RP_Top'], 
    #                     u1=1.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0,
    #                     amplitude='LoadAmplitude')
    model.DisplacementBC(name='LoadTopRP', createStepName='Step1',
                        region=assembly.sets['RP_Top'], 
                        u1=3.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)


    # Get all nodes from the instance
    all_nodes = instance.nodes

    # Get the node labels (IDs) for TopNodes and BottomNodes
    top_node_labels = [node.label for node in top_nodes]
    bottom_node_labels = [node.label for node in bottom_nodes]

    # Collect labels of intermediate nodes
    intermediate_node_labels = [node.label for node in all_nodes 
                                if node.label not in top_node_labels and node.label not in bottom_node_labels]

    # Use `sequenceFromLabels` to create a MeshNodeArray for intermediate nodes
    intermediate_nodes = instance.nodes.sequenceFromLabels(labels=intermediate_node_labels)

    # Create a node set for the intermediate nodes
    assembly.Set(name='IntermediateNodes', nodes=intermediate_nodes)

    # Restrict Z movement of intermediate nodes
    model.DisplacementBC(name='RestrictZIntermediateNodes', createStepName='Initial',
                        region=assembly.sets['IntermediateNodes'], 
                        u3=0.0, ur1=0.0, ur2=0.0)


    # Define the output directory for the current case
    output_dir = dir

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the CAE file to the specific directory
    cae_file_path = output_dir +'/{}/'.format(inp_name) +model_name + '.cae'
    mdb.saveAs(pathName=cae_file_path)

# Create analysis job
    job_name = inp_name   # Define job name
    job = mdb.Job(
        name=job_name, 
        model=model_name, 
        description="Job for Model",
        numCpus=4,
        numDomains=4
    )

    # Output inp file to the specific directory
    inp_file_path = output_dir +'/{}/'.format(inp_name) + job_name + '.inp'

    job.writeInput(consistencyChecking=OFF)  # Write the inp file to the default location

    # Move the inp file to the desired directory
    default_inp_path = output_dir +'/' + job_name + '.inp'
    if os.path.exists(default_inp_path):  # Check if the default inp file exists
            if os.path.exists(inp_file_path):  # Check if the target inp file exists
                os.remove(inp_file_path)  # Remove the existing target file
            os.rename(default_inp_path, inp_file_path)  # Move (rename) the inp file to the target directory

    print("CAE file saved to:", cae_file_path)
    print("INP file saved to:", inp_file_path)

    # 在每次循环结束时强制清理内存
    import gc
    gc.collect()
    del mdb.models[model_name]
    import time
    time.sleep(0.5)  # 允许系统回收资源
    # 清理临时对象
    for temp_sketch in [s for s in mdb.models.keys() if '__profile__' in s]:
        del mdb.models[temp_sketch]
    
    # 删除所有后缀为 .jnl 的文件
    inp_name=str(int(inp_name)-1)
    folder_path = dir + '/{}/'.format(inp_name)  # 确定目标文件夹路径

    try:
        for file_name in os.listdir(folder_path):  # 遍历文件夹中的所有文件
            if file_name.endswith('.jnl'):  # 检查文件是否以 .jnl 为后缀
                file_path = os.path.join(folder_path, file_name)  # 获取完整的文件路径
                print('Remove:',file_path)
                os.remove(file_path)  # 删除文件
    except WindowsError:pass

fileindex=1
startindex=1
for i in range(1, 21):
    dir='E:/LT/LatentDiffusion/PythonProject01/LatentDiffusion/Dataset_customer/Demo'
    inp_name='{}'.format(i)
    model_name = 'MODEL{}'.format(i)
    part_name = 'PART{}'.format(i)
    assembly_name = 'ASSEMBLY{}'.format(i)
    element_size = [0.390625,0.390625]  # Size of the square element in the xy plane
    # element_size = random.uniform(1.0, 2.0)
    thickness = 5.0  # Thickness of the solid elements in the z direction
    thickness_layers = 1  # Number of mesh layers in the thickness direction
    disp=3.0
    cutoff = [128,128] #y,x
    random_size_file_path = dir + '/{}/element_size.txt'.format(inp_name)
    with open(random_size_file_path, 'w') as f:
        f.write(str(element_size))
    func(dir,inp_name,model_name,part_name,assembly_name,element_size,thickness,thickness_layers,disp,cutoff,num_repeats=[1,1])
    dirsavetxt='E:/LT/LatentDiffusion/PythonProject01/LatentDiffusion/Dataset_customer/Demo/01.txt'
    with open(dirsavetxt, 'w') as f:
        f.write(str(i))