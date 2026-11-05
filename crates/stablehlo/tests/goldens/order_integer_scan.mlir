module {
  func.func @logdensity(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.constant dense<-2147483648> : tensor<i32>
    %2 = stablehlo.constant dense<-2147483648> : tensor<4xi32>
    %6:3 = stablehlo.while(%3 = %0, %4 = %1, %5 = %2) : tensor<i32>, tensor<i32>, tensor<4xi32>
    cond {
      %7 = stablehlo.constant dense<4> : tensor<i32>
      %8 = stablehlo.compare LT, %3, %7, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %8 : tensor<i1>
    } do {
      %9 = stablehlo.dynamic_slice %arg0, %3, sizes = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
      %10 = stablehlo.reshape %9 : (tensor<1xi32>) -> tensor<i32>
      %11 = stablehlo.maximum %4, %10 : tensor<i32>
      %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
      %13 = stablehlo.dynamic_update_slice %5, %12, %3 : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
      %14 = stablehlo.constant dense<1> : tensor<i32>
      %15 = stablehlo.add %3, %14 : tensor<i32>
      stablehlo.return %15, %11, %13 : tensor<i32>, tensor<i32>, tensor<4xi32>
    }
    %16 = stablehlo.constant dense<2147483647> : tensor<i32>
    %17 = stablehlo.constant dense<2147483647> : tensor<4xi32>
    %21:3 = stablehlo.while(%18 = %0, %19 = %16, %20 = %17) : tensor<i32>, tensor<i32>, tensor<4xi32>
    cond {
      %22 = stablehlo.constant dense<4> : tensor<i32>
      %23 = stablehlo.compare LT, %18, %22, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %23 : tensor<i1>
    } do {
      %24 = stablehlo.dynamic_slice %arg0, %18, sizes = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
      %25 = stablehlo.reshape %24 : (tensor<1xi32>) -> tensor<i32>
      %26 = stablehlo.minimum %19, %25 : tensor<i32>
      %27 = stablehlo.reshape %26 : (tensor<i32>) -> tensor<1xi32>
      %28 = stablehlo.dynamic_update_slice %20, %27, %18 : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
      %29 = stablehlo.constant dense<1> : tensor<i32>
      %30 = stablehlo.add %18, %29 : tensor<i32>
      stablehlo.return %30, %26, %28 : tensor<i32>, tensor<i32>, tensor<4xi32>
    }
    return %6#2, %21#2 : tensor<4xi32>, tensor<4xi32>
  }
}
