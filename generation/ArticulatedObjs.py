class ArticulatedObject():
    def __init__(self, class_id, geometry, parameters, xml, pose, rotation):
        self.type = class_id
        self.geom = geometry
        self.params = parameters
        self.pose = pose
        self.rotation = rotation
        self.xml = xml
        # self.cam_params = cam_params

class Microwave(ArticulatedObject):
    def __init__(self, class_id, geometry, parameters, xml, pose, rotation):
        super(Microwave, self).__init__(class_id, geometry, parameters, xml, pose, rotation)
        self.control = -0.2


class Drawer(ArticulatedObject):
    def __init__(self, class_id, geometry, parameters, xml, pose, rotation):
        super(Drawer, self).__init__(class_id, geometry, parameters, xml, pose, rotation)
        self.control = 0.2

class Toaster(ArticulatedObject):
    def __init__(self, class_id, geometry, parameters, xml, pose, rotation):
        super(Toaster, self).__init__(class_id, geometry, parameters, xml, pose, rotation)
        self.control = 0.2

class Cabinet(ArticulatedObject):
    def __init__(self, class_id, geometry, parameters, xml, pose, rotation):
        super(Cabinet, self).__init__(class_id, geometry, parameters, xml, pose, rotation)
        if self.geometry[3] == 1:
            self.control = -0.2
        else:
            self.control = 0.2
