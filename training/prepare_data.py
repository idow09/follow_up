import os

def parse_experiment_results(self, images_source, labels_root, preds_root):
    """
    :param images_source: Images root directory OR txt file containing paths
    :param labels_root: Labels root directory
    :param preds_root: Results (preds) root directory
    """
    if self.fake and self.persist:
        if os.path.exists(preds_root):
            shutil.rmtree(preds_root)
        os.makedirs(preds_root)

    image_paths = load_image_paths(images_source)
    for image_path in image_paths:
        sample_data = self.parse_sample_results(image_path, labels_root, preds_root)
        self.sample_list.append(sample_data)
    if len(self.sample_list) < 1:
        print('WARNING! No samples were found during parsing.')

project_root = 'C:\\Users\\dana\\Documents\\Ido\\follow_up_project'
data_root = os.path.join(project_root, 'datasets', 'efi')
images_root = os.path.join(data_root, 'images', 'good')
labels_root = os.path.join(data_root, 'labels', 'fused')
results_root =os.path.join(project_root, 'benchmark', '')

image_paths = load_image_paths(images_source)