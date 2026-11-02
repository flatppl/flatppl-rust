module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2x2xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = stablehlo.cholesky %arg0, lower = true : tensor<2x2xf32>
    %6 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %7 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %8 = stablehlo.compare EQ, %6, %7 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %9 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %10 = stablehlo.select %8, %5, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%10 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %13 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %15 = stablehlo.compare GT, %12, %14 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %16 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %17 = stablehlo.compare LT, %12, %16 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %18 = stablehlo.select %15, %17, %15 : (tensor<2xi1>, tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %19 = stablehlo.constant dense<true> : tensor<i1>
    %20 = stablehlo.reduce(%18 init: %19) applies stablehlo.and across dimensions = [0] : (tensor<2xi1>, tensor<i1>) -> tensor<i1>
    %21 = stablehlo.select %20, %2, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %22 = stablehlo.compare EQ, %4, %2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %23 = stablehlo.select %22, %21, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %24 = stablehlo.cholesky %arg2, lower = true : tensor<2x2xf32>
    %25 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %26 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %27 = stablehlo.compare EQ, %25, %26 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %28 = stablehlo.select %27, %24, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %29 = stablehlo.reduce(%28 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %30 = stablehlo.log %29 : tensor<2xf32>
    %31 = stablehlo.reduce(%30 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.constant dense<2.0> : tensor<f32>
    %33 = stablehlo.multiply %32, %31 : tensor<f32>
    %34 = stablehlo.log %12 : tensor<2xf32>
    %35 = stablehlo.reduce(%34 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %36 = stablehlo.multiply %32, %35 : tensor<f32>
    %37 = "stablehlo.triangular_solve"(%5, %24) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %38 = stablehlo.multiply %37, %37 : tensor<2x2xf32>
    %39 = stablehlo.reduce(%38 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %40 = stablehlo.reduce(%39 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %41 = stablehlo.constant dense<0.5> : tensor<f32>
    %42 = stablehlo.constant dense<3.0> : tensor<f32>
    %43 = stablehlo.subtract %arg1, %42 : tensor<f32>
    %44 = stablehlo.multiply %41, %43 : tensor<f32>
    %45 = stablehlo.multiply %44, %33 : tensor<f32>
    %46 = stablehlo.constant dense<-0.5> : tensor<f32>
    %47 = stablehlo.multiply %46, %40 : tensor<f32>
    %48 = stablehlo.multiply %arg1, %32 : tensor<f32>
    %49 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %50 = stablehlo.multiply %48, %49 : tensor<f32>
    %51 = stablehlo.multiply %46, %50 : tensor<f32>
    %52 = stablehlo.multiply %41, %arg1 : tensor<f32>
    %53 = stablehlo.negate %52 : tensor<f32>
    %54 = stablehlo.multiply %53, %36 : tensor<f32>
    %55 = stablehlo.constant dense<0.5723649429247001> : tensor<f32>
    %56 = stablehlo.add %52, %0 : tensor<f32>
    %57 = chlo.lgamma %56 : tensor<f32> -> tensor<f32>
    %58 = stablehlo.add %55, %57 : tensor<f32>
    %59 = stablehlo.add %52, %46 : tensor<f32>
    %60 = chlo.lgamma %59 : tensor<f32> -> tensor<f32>
    %61 = stablehlo.add %58, %60 : tensor<f32>
    %62 = stablehlo.negate %61 : tensor<f32>
    %63 = stablehlo.add %45, %47 : tensor<f32>
    %64 = stablehlo.add %63, %51 : tensor<f32>
    %65 = stablehlo.add %64, %54 : tensor<f32>
    %66 = stablehlo.add %65, %62 : tensor<f32>
    return %66, %23 : tensor<f32>, tensor<i32>
  }
}
