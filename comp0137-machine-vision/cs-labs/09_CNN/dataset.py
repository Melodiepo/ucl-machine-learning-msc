import torch
torch.manual_seed(42)
from PIL import Image

class CTDataset(torch.utils.data.Dataset):
    '''
    This utility corresponds to the Dataset class construction for the custom example of camera trap data.
    The class is responsible for how images and the corresponding data are loaded in batches during training or evaluation.
    '''   
    def __init__(self,annotation_dict,transform, base_path):
        
        ### TODO: Given annotation dictionary for a given split type (train,validation or test) and the selected set of transformations, 
        ### we initialize the properties of the class (img_paths, targets, species). Hint: loop through dictionary rows.
        
        self.transform = transform
        self.base_path = base_path
        
        # Initialize empty lists for img_paths, targets, species
        self.img_paths = []
        self.targets = []
        self.species = []
        
        # Loop through annotation dictionary to populate the lists
        for row in annotation_dict:
            self.img_paths.append(f"{self.base_path}/{row[0]}")
            self.targets.append(row[1])
            self.species.append(row[2])

        ## TODO: Fill properly the values in the three attribute lists (img_paths, targets, species) defined below
        ## self.img_paths=['train/59c80554-23d2-11e8-a6a3-ec086b02610b_0_opossum.jpg']
        ## self.targets=[0]
        ## self.species=['opossum']
        
                
    def __len__(self):
        return len(self.img_paths)
    
    def get_image(self,img_path):  

        with open(img_path, 'rb') as f:
            try:
                img = Image.open(f)
                return img.convert('RGB')
            except:
                print("Image from {} cannot be read. It will be skipped".format(img_path))
                return None
            
    def __getitem__(self, idx):        
        ### TODO: Given an index we want to return the corresponding dictionary (op) with 'img_path','target','species' as its keys, along with 
        ### 'img' that corresponds to the transformed image. This routine is typically called at a batch level during train/validation.
        """
        Given an index, return a dictionary containing the following:
        - 'img_path': Path to the image
        - 'target': Target label for the image
        - 'species': Species name for the image
        - 'img': Transformed image
        """
        op = {}
        
        # Retrieve the corresponding values for the given index
        op['img_path'] = self.img_paths[idx]
        op['target'] = self.targets[idx]
        op['species'] = self.species[idx]
        
        # Load and transform the image
        img = self.get_image(op['img_path'])
        if img is not None:
            op['img'] = self.transform(img)
        else:
            # If the image cannot be loaded, set the 'img' key to None
            op['img'] = None
        
       ### TODO: Comment out the following placeholder commands that load values in the dictionary to be returned 
       ### (the existing lines return the same image always!) and replace them with the corresponding values to be loaded at every batch.
        ##op['target'] = 0
        ##op['img_path'] = 'train/59c80554-23d2-11e8-a6a3-ec086b02610b_0_opossum.jpg'
        ##op['species'] = 'opossum'
        ##op['img'] =  self.transform(self.get_image(op['img_path']))
        return op
    
    