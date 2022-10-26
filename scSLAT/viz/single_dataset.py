r"""
Vis simulated dataset which know ground truth
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class match_3D():
    r"""
    Plot the mapping result and compare with ground truth in 3D 
    
    Parameters
    -----------
    slices
        List of spatial coordinates
    matching
        Matching pairs
    matching_info
         matching correctness, output of `metrics.hit_k()`
    subsample_size
        Subsample size of matches
    cell_types 
        pandas which index is ['dataset1', 'dataset2']
    k
        top k-th hit 
        
    Note
    -----------
    Use `xarray` to avoid chaos in data and meta
        
    """
    def __init__(self,slices: List[np.ndarray], 
                 matching: np.ndarray,
                 matching_info: pd.DataFrame=None,
                 cell_types: Optional[List[str]]=None,
                 subsample_size: Optional[int]=100,
                 k: Optional[int]=10
    ) -> None:
        self.len = len(slices)
        self.slices = slices
        self.cell_types = cell_types
        self.matching_info = matching_info
        self.k = k
        
        assert self.len == 2
        assert slices[0].shape[1] == matching.shape[1]
        if matching.shape[1] < 100:
            self.matching = matching
        else:
            assert matching.shape[1] > subsample_size
            print(f'Just subsample {subsample_size} cell pairs from {matching.shape[1]}')
            self.matching = matching[:,np.random.choice(matching.shape[1],subsample_size, replace=False)]   
    
    # def manage_data() -> xr.DataArray:
        # data -> spatial; 
        # data = xr.DataArray()
    
    def draw_3D(self, figure_size:List[int]=[12,12]) -> None:
        r"""
        Draw 3D picture of two datasets
        
        Parameters:
        -----------
        size
            plt figure size
        figure_size
            figure size [length * width]
        """
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection='3d')
        # plot by different cell types
        if isinstance(self.cell_types,pd.DataFrame):
            cell_types = set(self.cell_types.iloc[:,0].unique())
            color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(cell_types))]
            c_map = {}
            for i, celltype in enumerate(cell_types):
                c_map[celltype] = color[i]
            for cell_type in cell_types:
                sub_slices = self.slices[:,:,(self.cell_types['dataset1'] == cell_type)&(self.cell_types['dataset2'] == cell_type)]
                for i in range(self.len):
                    slice = sub_slices[i]
                    xs = slice[0]
                    ys = slice[1]
                    zs = i
                    ax.scatter(xs,ys,zs,s=0.1,c=c_map[cell_type])
                    
        # plot different point layers
        else:
            for i in range(self.len):
                slice = self.slices[i]
                xs = slice[0]
                ys = slice[1]
                zs = i
                ax.scatter(xs,ys,zs,s=0.1)
        
        # plot line
        self.draw_lines(self.slices,self.matching,ax)
        
        plt.axis('off')
        plt.show()
        
    def draw_lines(self, slices, matching:np.ndarray,ax) -> None:
        r"""
        Draw lines between paired cells in two datasets
                
        Parameters:
        -----------
        slices
            List of spatial coordinates
        matching
            Matching pairs
        """
        for i in range(matching.shape[1]):
            color = 'grey'
            if isinstance(self.matching_info,pd.DataFrame) and self.matching_info['h1'][i]:
                color = '#ade8f4'
            elif isinstance(self.matching_info,pd.DataFrame) and self.matching_info[f'h{self.k}'][i]:
                    color = '#588157'
            else:
                color = '#ffafcc'
            pair = matching[:,i]
            point0 = np.append(slices[0][:,pair[0]],0)
            point1 = np.append(slices[1][:,pair[1]],1)
            coord = np.row_stack((point0,point1))
            ax.plot(coord[:,0], coord[:,1], coord[:,2], color = color, linestyle="dashed",linewidth=0.3)
            