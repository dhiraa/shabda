# import configparser
# from configparser import ExtendedInterpolation
# from shabda.helpers.print_helper import *
#
#
# class ConfigManager(object):
#     def __init__(self, config_path=None, key_value_pair_dict=None):
#         # set the path to the _config file
#
#         #TODO As we need to read nested sections we might need to use ConfigObj instead of Config Parser
#         #self.config = configobj.ConfigObj(interpolation=ExtendedInterpolation())
#
#         self.config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
#         self.config_path = config_path
#         self.key_value_pair_dict = key_value_pair_dict
#
#         if self.config_path is not None:
#             print_info("Reading _config from : " + config_path)
#             self.config.read(self.config_path)
#         elif self.key_value_pair_dict is not None:
#             print_info("Reading _config from dict")
#             self.config.read_dict(dictionary=self.key_value_pair_dict)
#         else:
#             raise AssertionError("Either config file path or Key Value string should be given")
#
#     def get_sections(self):
#         return self.config._sections.keys()
#
#     def set_item(self, section, option, value):
#         self.config.set(section=section,
#                         option=option,
#                         value=value)
#
#     def get_item(self, section, option)-> str:
#         return self.config.get(section=section,
#                                option=option)
#
#     def add_section(self, section):
#         self.config.add_section(section)
#
#     def get_item_as_float(self,section, option):
#         return self.config.getfloat(section=section,
#                                option=option)
#
#     def get_item_as_int(self,section, option):
#         return self.config.getint(section=section,
#                                option=option)
#
#     def get_item_as_boolean(self,section, option):
#         return self.config.getboolean(section=section,
#                                option=option)
#
#     def save_config(self):
#         # Writing our configuration file to '_config'
#         with open(self.config_path, 'w') as configfile:
#             self.config.write(configfile)
