# python的单元测试框架很多unittest、pytest、nose等
import unittest,sys
import pytest,nose
import HTMLTestRunner


def testfun(a,b):
	return a+b

class Demo(unittest.TestCase):
	# 测试函数需test开头，可以使用断言函数
	def setUp(self):
		pass #初始化函数相当init

	def test_add_4_5(self):
		self.assertEqual(testfun(4,5),9)
	def test_add_4_6(self):
		self.assertEqual(testfun(4,6),10)

	def tearDown(self):
		pass #退出时函数这里可文件关闭 socket关闭

if __name__ == '__main__':
	# unittest.main()
	# 在命令行模式下可使用-v和-f等命令输出

	# 默认txt报告使用模块生成html报告
	suite = unittest.TestSuite()
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(Demo))
	runner = HTMLTestRunner.HTMLTestRunner(open("result.html",'w'))
	runner.run(suite)

	# 改了个html模块的bug，解码问题
	strs = "this is string example....wow!!!";
	strs = strs.encode('gbk','strict');
	print(type(strs),strs)
