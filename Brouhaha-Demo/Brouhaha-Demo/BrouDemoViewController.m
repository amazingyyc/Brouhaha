/**
 * the entrance of demo
 *
 * Created by yanyuanchi on 2017/8/13.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#import "BrouDemoViewController.h"
#import "LeNetViewController.h"
#import "ArtTransformViewController.h"

@interface BrouDemoViewController () <UITableViewDataSource, UITableViewDelegate>

@property (nonatomic, strong) UITableView *tableView;

@end

@implementation BrouDemoViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    CGRect bounds = self.view.bounds;
    
    _tableView = [[UITableView alloc] initWithFrame:CGRectMake(0, 20, bounds.size.width, bounds.size.height - 20)
                                              style:UITableViewStylePlain];
    _tableView.dataSource = self;
    _tableView.delegate   = self;
    
    [_tableView registerClass:[UITableViewCell class] forCellReuseIdentifier:@"cellId"];
    
    [self.view addSubview:_tableView];
    [self.view setBackgroundColor:[UIColor whiteColor]];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    if (0 == section) {
        return 2;
    }
    
    return 0;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [_tableView dequeueReusableCellWithIdentifier:@"cellId"];
    
    if (0 == indexPath.row) {
        cell.textLabel.text = @"LeNet Demo";
    } else if (1 == indexPath.row) {
        cell.textLabel.text = @"Artistic Style Transform Demo";
    }
    
    return cell;
}

- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath {
    return 100;
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];// 取消选中
    
    if (0 == indexPath.row) {
        LeNetViewController *ctrl = [[LeNetViewController alloc] init];
        
        [self presentViewController:ctrl animated:YES completion:nil];
    } else {
        ArtTransformViewController *ctrl = [[ArtTransformViewController alloc] init];
        
        [self presentViewController:ctrl animated:YES completion:nil];
    }
}

@end









