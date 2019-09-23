import NGTS_Field
import NGTS_Object

fieldname = 'NG2331-3922'
root_dir = '/data/jtb34'

field = NGTS_Field.return_field_from_object_directory(root_dir, fieldname)
field.plot_objects_vs_period()

